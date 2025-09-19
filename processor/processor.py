import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from torch.cuda import amp
import torch.distributed as dist

def reduce_tensor(tensor, world_size):
    """
    Reduce tensor across all GPUs and return the average
    """
    if world_size == 1:
        return tensor
    
    # Clone to avoid modifying original tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_img_loader,
             val_txt_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    
    # 确保已经初始化了分布式环境
    if cfg.MODEL.DIST_TRAIN:
        # 获取全局进程数和工作组大小
        world_size = dist.get_world_size()
        logger.info(f'World size: {world_size}')

    model.to(local_rank)

    # 如果使用分布式训练，包装模型
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
        logger.info('Using DistributedDataParallel for training')
        
        # 使用更优化的配置
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=True,
            gradient_as_bucket_view=True   # 可以提高性能
        )

    # 初始化计量器
    loss_meter = AverageMeter()
    smi_meter = AverageMeter()
    itc_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_text = AverageMeter()
    itm_meter = AverageMeter()
    # acc_clot_meter = AverageMeter()

    # 初始化评估器和梯度缩放器
    evaluator = Evaluator(val_img_loader, val_txt_loader)
    scaler = amp.GradScaler()

    # 训练循环
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        smi_meter.reset()
        itc_meter.reset()
        acc_meter.reset()
        acc_text.reset()
        itm_meter.reset()
        # acc_clot_meter.reset()

        model.train()
        n_iter_overall = 0
        for n_iter, (img, instruction, vid, target_cam, target_view ) in enumerate(train_loader):
      
            n_iter_overall += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            # instruction = instruction.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                # batch = img.size(0)
                # instruction = ('do_not_change_clothes',) * batch
                # score, feat, _ = model(img, instruction, label=target, cam_label=target_cam, view_label=target_view )
                outputs = model(img, instruction, label=target, cam_label=target_cam, view_label=target_view)
                feat, bio_f, clot_f, score, f_logits, c_logits, local_feat_all, text_embeds_s, text_score, loss_itm, loss_itc = outputs
                
                # 修改损失函数调用，确保所有输出都参与计算
                loss, smi_loss = loss_fn(
                    score, f_logits, c_logits, feat, bio_f, clot_f, 
                    target, text_embeds_s, text_score, target_cam, epoch,
                    local_feat_all, loss_itm, loss_itc  # 添加之前未使用的输出
                )
                
                # 移除之前的手动正则化，因为现在损失函数会处理所有输出
                # if local_feat_all is not None and torch.is_tensor(local_feat_all):
                #     regularization_loss = (
                #         local_feat_all.pow(2).mean() * 1e-8 +
                #         bio_f.pow(2).mean() * 1e-8 +
                #         clot_f.pow(2).mean() * 1e-8
                #     )
                #     loss = loss + regularization_loss
                
                # 注意：loss_itm 和 loss_itc 现在已经在损失函数内部处理
                # 移除这行重复添加: loss += loss_itm*10 + loss_itc*6
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
                acc_caption = (text_score.max(1)[1] == target).float().mean()
                acc_bio = (f_logits.max(1)[1] == target).float().mean()
                acc_clot = (c_logits.max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
                acc_caption = (text_score.max(1)[1] == target).float().mean()
                acc_bio = (f_logits.max(1)[1] == target).float().mean()
                acc_clot = (c_logits.max(1)[1] == target).float().mean()

            # 同步损失值用于准确的日志记录
            if cfg.MODEL.DIST_TRAIN:
                # 将损失转换为tensor进行同步
                loss_tensor = torch.tensor(loss.item()).cuda()
                smi_loss_tensor = torch.tensor(smi_loss.item()).cuda()
                itc_loss_tensor = torch.tensor(loss_itc.item()).cuda()
                itm_loss_tensor = torch.tensor(loss_itm.item()).cuda()
                
                # 同步所有GPU的损失
                world_size = dist.get_world_size()
                loss_avg = reduce_tensor(loss_tensor, world_size).item()
                smi_loss_avg = reduce_tensor(smi_loss_tensor, world_size).item()
                itc_loss_avg = reduce_tensor(itc_loss_tensor, world_size).item()
                itm_loss_avg = reduce_tensor(itm_loss_tensor, world_size).item()
                
                loss_meter.update(loss_avg, img.shape[0])
                smi_meter.update(smi_loss_avg, img.shape[0])
                itc_meter.update(itc_loss_avg, img.shape[0])
                itm_meter.update(itm_loss_avg, 1)
            else:
                loss_meter.update(loss.item(), img.shape[0])
                smi_meter.update(smi_loss.item(), img.shape[0])
                itc_meter.update(loss_itc.item(), img.shape[0])
                itm_meter.update(loss_itm, 1)

            acc_meter.update(acc, 1)
            acc_text.update(acc_caption, 1)
            # acc_clot_meter.update(acc_clot, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                # 让每个进程都输出日志，添加rank标识
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    logger.info("Rank[{}/{}] Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Acc_text: {:.3f}, itm_loss: {:.3f}, smi_Loss: {:.3f}, itc_loss: {:.3f}, Base Lr: {:.2e}"
                                .format(rank, world_size-1, epoch, (n_iter + 1), len(train_loader), 
                                       loss_meter.avg, acc_meter.avg, acc_text.avg, itm_meter.avg, 
                                       smi_meter.avg, itc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Acc_text: {:.3f}, itm_loss: {:.3f}, smi_Loss: {:.3f}, itc_loss: {:.3f},  Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, acc_text.avg, itm_meter.avg,  smi_meter.avg, itc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter_overall + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            # 每个进程都输出epoch完成信息
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            total_batch_size = train_loader.batch_size * world_size
            logger.info("Rank[{}/{}] Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(rank, world_size-1, epoch, time_per_batch * (n_iter_overall + 1), 
                           total_batch_size / time_per_batch))
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter_overall + 1), train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 每个进程都保存模型（但使用不同文件名避免冲突）
                rank = dist.get_rank()
                if rank == 0:  # 只让主进程保存最终模型
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch < 5:
            if cfg.MODEL.DIST_TRAIN:
                # 只让主进程进行验证，避免重复计算
                rank = dist.get_rank()
                if rank == 0:
                    model.eval()
                    logger.info("Rank[{}] Starting validation for Epoch: {}".format(rank, epoch))
                    
                    with torch.no_grad():
                        t2i_Rank1, t2i_Rank5, t2i_Rank10, t2i_mAP, t2i_mINP = evaluator.eval(model.eval())
                        logger.info("Rank[{}] Validation completed for Epoch: {}".format(rank, epoch))
                        logger.info("Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1f}".format(t2i_mAP))
                        logger.info("mINP: {:.1f}".format(t2i_mINP))
                        logger.info("Rank-1: {:.1f}".format(t2i_Rank1))
                        logger.info("Rank-5: {:.1f}".format(t2i_Rank5))
                        logger.info("Rank-10: {:.1f}".format(t2i_Rank10))
                    
                    torch.cuda.empty_cache()
                else:
                    # 其他进程等待主进程完成验证
                    logger.info("Rank[{}] Waiting for validation to complete...".format(rank))
                
                # 同步所有进程，确保验证完成后再继续
                dist.barrier()
            else:
                with torch.no_grad():
                    logger.info("Starting validation for Epoch: {}".format(epoch))
                    t2i_Rank1, t2i_Rank5, t2i_Rank10, t2i_mAP, t2i_mINP = evaluator.eval(model.eval())
                    logger.info("Validation completed for Epoch: {}".format(epoch))
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1f}".format(t2i_mAP))
                    logger.info("mINP: {:.1f}".format(t2i_mINP))
                    logger.info("Rank-1: {:.1f}".format(t2i_Rank1))
                    logger.info("Rank-5: {:.1f}".format(t2i_Rank5))
                    logger.info("Rank-10: {:.1f}".format(t2i_Rank10))
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

    for n_iter, (img, instruction, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # feat , _ = model(img, cam_label=camids, view_label=target_view)
            batch = img.size(0)
            # = ('do_not_change_clothes',) * batch TODO
            feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s = model(img, instruction,  cam_label=camids, view_label=target_view )
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


