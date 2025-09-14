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
    # eval_period = 5#TODO

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
    smi_meter = AverageMeter()
    itc_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_text = AverageMeter()
    itm_meter = AverageMeter()
    # acc_clot_meter = AverageMeter()

    evaluator = Evaluator(val_img_loader, val_txt_loader)
    scaler = amp.GradScaler()
    # train

    #TODO 写了个假的instruction
    
    
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
                feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s, text_score, loss_itm, loss_itc= model(img, instruction, label=target, cam_label=target_cam, view_label=target_view )
                loss, smi_loss= loss_fn(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, text_score, target_cam, epoch)
                # if epoch > 20:
                loss += loss_itm*10 + loss_itc*6
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

            loss_meter.update(loss.item(), img.shape[0])
            smi_meter.update(smi_loss.item(), img.shape[0])
            itc_meter.update(loss_itc.item(), img.shape[0])
            acc_meter.update(acc, 1)
            acc_text.update(acc_caption, 1)
            itm_meter.update(loss_itm, 1)
            # acc_clot_meter.update(acc_clot, 1)

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
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Acc_text: {:.3f}, itm_loss: {:.3f}, smi_Loss: {:.3f}, itc_loss: {:.3f},  Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, acc_text.avg, itm_meter.avg,  smi_meter.avg, itc_meter.avg, base_lr))

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

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch < 5:
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
                with torch.no_grad():
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    t2i_Rank1, t2i_Rank5, t2i_Rank10, t2i_mAP, t2i_mINP = evaluator.eval(model.eval())
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


