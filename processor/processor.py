import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss.Mentor import MentorNet, mixup_data, sigmoid

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    # eval_period = cfg.SOLVER.EVAL_PERIOD
    eval_period = 1#TODO
    burn_in_epoch = 15  #TODO 前 15 个 epoch 用于计算 MentorNet 权重
    loss_moving_avg = 0.0  # 在循环开始之前初始化滑动平均损失
    loss_moving_average_decay = 0.5  # 滑动平均损失的衰减率
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model.to(local_rank)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mentornet = MentorNet(label_embedding_size=8,
                        epoch_embedding_size=6, 
                        num_label_embedding=751,
                        num_fc_nodes=100).to('cuda')
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train

    #TODO 写了个假的instruction
    
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        n_iter_overall = 0
        epochs = torch.tensor([epoch], dtype=torch.int32)

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            n_iter_overall += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # cur_epoch = min(epoch, burn_in_epoch)  # 使用 min 来代替 tf.minimum
            # cur_epoch = torch.tensor(cur_epoch, dtype=torch.int32)  # 转换为 int32 类型
            with amp.autocast(enabled=True):
                batch = img.size(0)
                instruction = ('do_not_change_clothes',) * batch

                # --------- 计算原始 loss -------------
                feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s = model(
                    img, instruction, label=target, cam_label=target_cam, view_label=target_view
                )
                loss = loss_fn(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, target_cam)

                # --------- MentorNet 计算权重 v -------------
                loss_reshaped = loss.view(-1, 1)  # (batch, 1)
                epoch_tensor = torch.full((batch, 1), fill_value=epoch, dtype=torch.float32).to(device)  # (batch, 1) 当前 epoch

                # 计算 loss 分位数（percentile_loss）
                percentile_loss = torch.quantile(loss.detach(), q=0.3)  # 取 20% 分位损失
                loss_moving_avg = loss_moving_average_decay  * loss_moving_avg + (1 - loss_moving_average_decay)  * percentile_loss  # 更新滑动平均损失
                lossdiff = loss - loss_moving_avg  # 计算损失与滑动均值的差异

                # 生成 v 的上下界
                v_ones = torch.ones_like(loss, dtype=torch.float32)
                v_zeros = torch.zeros_like(loss, dtype=torch.float32)

                # # 计算 upper_bound，类似于 tf.cond()
                # upper_bound = torch.where(epoch < (burn_in_epoch - 1), v_ones, v_zeros)

                # 根据 cur_epoch 和 burn_in_epoch 来选择 v_ones 或 v_zeros
                # if cur_epoch < (burn_in_epoch - 1):
                upper_bound = v_ones
                # else:
                #     upper_bound = v_zeros

                # MentorNet 计算 v
                input_data = torch.cat([loss_reshaped, lossdiff.unsqueeze(1), target.unsqueeze(1), epoch_tensor], dim=1).to('cuda')  # 拼接输入
                v = sigmoid(mentornet(input_data)) # MentorNet 计算权重 (batch, 1)
                if isinstance(v, np.ndarray):
                    v = torch.tensor(v)  # 转换为 Tensor

                v = v.to(device)
                upper_bound = upper_bound.to(device)

                v = torch.maximum(v, upper_bound)  # 限制 v 的最大值

                # 1. 阻断 v 的梯度
                v = v.detach()  # v 的梯度被阻断，不会在反向传播中计算

                # 2. 加权损失
                weighted_loss_vector = loss * v  # 对每个样本的损失进行加权

                # 3. 计算加权损失的平均值
                loss = weighted_loss_vector.mean()  # 返回加权损失的平均值作为最终损失

                # # --------- Mixup 数据增强 -------------
                # mixed_img, mixed_target = mixup_data(img, target, v)  # 使用 MentorNet 权重做 Mixup
                # feat_mix, bio_f_mix, clot_f_mix, score_mix, f_logits_mix, c_logits_mix, _, text_embeds_s_mix = model(
                #     mixed_img, instruction, label=mixed_target, cam_label=target_cam, view_label=target_view
                # )

                # loss_mixup = loss_fn(score_mix, f_logits_mix, c_logits_mix, feat_mix, bio_f_mix, clot_f_mix, mixed_target, text_embeds_s_mix, target_cam)

            # --------- 计算总损失并反向传播 -------------
            # loss_final = loss + loss_mixup
            scaler.scale(loss.sum()).backward()
            # print('loss:', loss.sum())
            scaler.step(optimizer)
            scaler.update()


            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.sum().item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter_overall + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter_overall + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s = model(img, instruction, label=target, cam_label=target_cam, view_label=target_view )
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        batch = img.size(0)
                        instruction = ('do_not_change_clothes',) * batch
                        # feat, _ = model(img, cam_label=camids, view_label=target_view)
                        feat, bio_f, clot_f, f_logits, c_logits, _, text_embeds_s = model(img, instruction, cam_label=camids, view_label=target_view )
                        bio_clot_feat = torch.cat([bio_f, clot_f], dim=1)
                        evaluator.update((bio_clot_feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # feat , _ = model(img, cam_label=camids, view_label=target_view)
            batch = img.size(0)
            instruction = ('do_not_change_clothes',) * batch
            feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s = model(img, instruction,  cam_label=camids, view_label=target_view )
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


