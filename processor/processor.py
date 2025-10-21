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


def add_parameter_hooks(model):
    """
    为模型参数添加钩子，确保所有参数都有梯度
    """
    hooks = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 使用更温和的方式确保参数参与计算
            def make_hook(param_name):
                def hook_fn(grad):
                    if grad is None:
                        # 如果梯度为None，创建一个很小的梯度
                        return torch.zeros_like(param) + 1e-12
                    elif grad.abs().sum() == 0:
                        # 如果梯度全为0，添加很小的扰动
                        return grad + 1e-12 * torch.randn_like(grad)
                    return grad
                return hook_fn
            
            hook = param.register_hook(make_hook(name))
            hooks.append((name, hook))
    
    return hooks


def remove_parameter_hooks(hooks):
    """
    移除参数钩子
    """
    for name, hook in hooks:
        hook.remove()


def check_unused_parameters(model, logger):
    """
    检查未使用的参数
    """
    # 支持 model 为 DistributedDataParallel 的情况
    core_model = model.module if hasattr(model, 'module') else model

    unused_params = []
    zero_grad_params = []
    
    for name, param in core_model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                unused_params.append(name)
            elif param.grad.abs().sum().item() == 0:
                zero_grad_params.append(name)
    
    if unused_params:
        logger.warning(f"参数没有梯度: {len(unused_params)} 个参数")
        # 只打印前5个，避免日志过长
        if len(unused_params) <= 5:
            logger.warning(f"详细列表: {unused_params}")
        else:
            logger.warning(f"前5个: {unused_params[:5]} ... (共{len(unused_params)}个)")
    
    if zero_grad_params:
        logger.warning(f"参数梯度为零: {len(zero_grad_params)} 个参数")
        if len(zero_grad_params) <= 5:
            logger.warning(f"详细列表: {zero_grad_params}")
        else:
            logger.warning(f"前5个: {zero_grad_params[:5]} ... (共{len(zero_grad_params)}个)")
    
    return len(unused_params), len(zero_grad_params)

# 在模型定义中的适当位置添加
def freeze_unnecessary_params(model):
    # 冻结文本编码器的特定层
    # 未使用的参数: ['module.text_encoder.cls.predictions.bias', 'module.text_encoder.cls.predictions.transform.dense.weight', 'module.text_encoder.cls.predictions.transform.dense.bias', 'module.text_encoder.cls.predictions.transform.LayerNorm.weight', 'module.text_encoder.cls.predictions.transform.LayerNorm.bias', 'module.base.norm0.weight', 'module.base.norm0.bias', 'module.base.norm1.weight', 'module.base.norm1.bias', 'module.base.norm2.weight', 'module.base.norm2.bias', 'module.base.semantic_embed_w.0.weight', 'module.base.semantic_embed_w.0.bias', 'module.base.semantic_embed_w.1.weight', 'module.base.semantic_embed_w.1.bias', 'module.base.semantic_embed_w.2.weight', 'module.base.semantic_embed_w.2.bias', 'module.base.semantic_embed_w.3.weight', 'module.base.semantic_embed_w.3.bias', 'module.base.semantic_embed_b.0.weight', 'module.base.semantic_embed_b.0.bias', 'module.base.semantic_embed_b.1.weight', 'module.base.semantic_embed_b.1.bias', 'module.base.semantic_embed_b.2.weight', 'module.base.semantic_embed_b.2.bias', 'module.base.semantic_embed_b.3.weight', 'module.base.semantic_embed_b.3.bias', 'module.bottleneck.weight', 'module.bottleneck.bias', 'module.fusion_feat_bn.weight', 'module.fusion_feat_bn.bias', 'module.feat_bn.weight', 'module.feat_bn.bias', 'module.vision_proj.0.weight', 'module.vision_proj.0.bias', 'module.vision_proj.2.weight', 'module.vision_proj.2.bias', 'module.text_proj.0.weight', 'module.text_proj.0.bias', 'module.text_proj.2.weight', 'module.text_proj.2.bias']
    patterns = [
        'text_encoder.cls.predictions',
        'base.norm',
        'base.semantic_embed_w',
        'base.semantic_embed_b',
        'bottleneck',
        'fusion_feat_bn',
        'feat_bn',
        'vision_proj',
        'text_proj'
    ]
    for name, param in model.named_parameters():
        for p in patterns:
            if p in name:
                param.requires_grad = False
    
    return model


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
        rank = dist.get_rank()
        if rank == 0:  # 只在卡0上打印
            logger.info('start training')
            logger.info(f'World size: {world_size}')
    else:
        logger.info('start training')

    model = freeze_unnecessary_params(model)
    model.to(local_rank)
    param_hooks = None

    # 如果使用分布式训练，包装模型
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        rank = dist.get_rank()
        if rank == 0:  # 只在卡0上打印
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
        # param_hooks = add_parameter_hooks(model)
        # if rank == 0:
        #     logger.info(f"Added hooks to {len(param_hooks)} parameters")

    # model = model.module if hasattr(model, 'module') else model

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
        # 对分布式采样器设置 epoch，保证不同 epoch 的随机性 & 各 rank 同步随机序列
        if cfg.MODEL.DIST_TRAIN and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            try:
                train_loader.sampler.set_epoch(epoch)
            except Exception as e:
                rank = dist.get_rank()
                if rank == 0:
                    logger.warning(f"设置 sampler epoch 失败: {e}")
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

            # if n_iter % (log_period * 5) == 0:
            if cfg.MODEL.DIST_TRAIN:
                rank = dist.get_rank()
                if rank == 0:  # 只在卡0上检查并打印
                    unused_count, zero_grad_count = check_unused_parameters(model, logger)
                    if unused_count > 0 or zero_grad_count > 0:
                        logger.warning(f"Epoch {epoch}, Iter {n_iter}: {unused_count} unused, {zero_grad_count} zero-grad parameters")
            else:
                unused_count, zero_grad_count = check_unused_parameters(model, logger)
                if unused_count > 0 or zero_grad_count > 0:
                    logger.warning(f"Epoch {epoch}, Iter {n_iter}: {unused_count} unused, {zero_grad_count} zero-grad parameters")


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
            else:
                acc = (score.max(1)[1] == target).float().mean()
                acc_caption = (text_score.max(1)[1] == target).float().mean()

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

            # 使用 batch 大小作为权重进行加权平均，使统计更准确
            batch_size = img.size(0)
            acc_meter.update(acc.item(), batch_size)
            acc_text.update(acc_caption.item(), batch_size)
            # acc_clot_meter.update(acc_clot, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                base_lr = scheduler.get_lr()[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                
                if epoch == 1 and (n_iter + 1) == log_period:
                    # Debug：打印前几个 pid 以验证不同 rank 取样是否不同（只打印一次即可）
                    # logger.info(f"[DEBUG] sample vid head: {vid[:8].tolist()}")
                    rank = dist.get_rank() 
                    if rank == 0:  # 只在卡0上打印
                        u = vid.unique().tolist()
                        logger.info(f"[DEBUG][Rank {rank}] first batch unique pids ({len(u)}): {u[:12]}")

                if cfg.MODEL.DIST_TRAIN:
                    rank = dist.get_rank()
                    if rank == 0:  # 只在卡0上打印
                        if epoch == 1 and (n_iter + 1) == log_period:
                            # Debug：打印前几个 pid 以验证不同 rank 取样是否不同（只打印一次即可）
                            u = vid.unique().tolist()
                            logger.info(f"[DEBUG][Rank {rank}] first batch unique pids ({len(u)}): {u[:12]}")
                        
                        world_size = dist.get_world_size()
                        logger.info(
                            "Rank[{}/{}] Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Acc_text: {:.3f}, itm_loss: {:.3f}, smi_Loss: {:.3f}, itc_loss: {:.3f}, Base Lr: {:.2e}".format(
                                rank, world_size-1, epoch, (n_iter + 1), len(train_loader),
                                loss_meter.avg, acc_meter.avg, acc_text.avg, itm_meter.avg,
                                smi_meter.avg, itc_meter.avg, base_lr
                            )
                        )
                else:
                    logger.info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Acc_text: {:.3f}, itm_loss: {:.3f}, smi_Loss: {:.3f}, itc_loss: {:.3f},  Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, acc_text.avg, itm_meter.avg, smi_meter.avg, itc_meter.avg, base_lr
                        )
                    )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter_overall + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            # 只在卡0上输出epoch完成信息
            rank = dist.get_rank()
            if rank == 0:
                world_size = dist.get_world_size()
                total_batch_size = train_loader.batch_size * world_size
                logger.info("Rank[{}/{}] Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(rank, world_size-1, epoch, time_per_batch * (n_iter_overall + 1), 
                               total_batch_size / time_per_batch))
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter_overall + 1), train_loader.batch_size / time_per_batch))

        if epoch % 10 == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 每个进程都保存模型（但使用不同文件名避免冲突）
                rank = dist.get_rank()
                if rank == 0:  # 只让主进程保存最终模型
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if 0 == 0:
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

    # 训练结束后移除所有钩子
    if param_hooks:
        if cfg.MODEL.DIST_TRAIN:
            rank = dist.get_rank()
            if rank == 0:
                logger.info("Removing parameter hooks...")
        else:
            logger.info("Removing parameter hooks...")
        remove_parameter_hooks(param_hooks)

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


