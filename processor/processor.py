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
    eval_period = cfg.SOLVER.EVAL_PERIOD
    # eval_period = 1#TODO
    loss_moving_avg = 0.0  # 在循环开始之前初始化滑动平均损失
    loss_moving_average_decay = 0.9  # 滑动平均损失的衰减率
    # loss_moving_average_decay = 0.3
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
                        num_label_embedding=767,
                        num_fc_nodes=100).to('cuda')
    loss_p_percentile = list(torch.linspace(30, 90, steps=121))
    burn_in_epoch = 110#12到24
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
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
    

        for n_iter, (img, instruction, vid, target_cam, target_view) in enumerate(train_loader):
            n_iter_overall += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # print('instruction:', instruction)    
            with amp.autocast(enabled=True):
                batch = img.size(0)
                # instruction = ('do_not_change_clothes',) * batch

                # --------- 计算原始 loss -------------
                feat, bio_f, clot_f, score, f_logits, c_logits, _, text_embeds_s = model(
                    img, instruction, label=target, cam_label=target_cam, view_label=target_view
                )
                loss = loss_fn(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, target_cam).to(device)

                # --------- MentorNet 计算权重 v -------------
                # loss_reshaped = loss.view(-1, 1)  # (batch, 1)


                # 计算 loss 分位数（percentile_loss）
                current_p = float(loss_p_percentile[epoch].to(device))
                percentile_loss = torch.quantile(loss.detach(), q=current_p/ 100.0 ).to(device)  # 取 20% 分位损失
                # percentile_loss = torch.quantile(loss.detach(), q=0.3 ).to(device)  # 取 20% 分位损失
                loss_moving_avg = loss_moving_average_decay  * loss_moving_avg + (1 - loss_moving_average_decay)  * percentile_loss  # 更新滑动平均损失
                lossdiff = loss - loss_moving_avg.to(device)  # 计算损失与滑动均值的差异

                # 生成 v 的上下界
                v_ones = torch.ones_like(loss, dtype=torch.float32)
                v_zeros = torch.zeros_like(loss, dtype=torch.float32)

                # 确保 epoch 和 burn_in_epoch 是张量
                epoch_tensor = torch.tensor(epoch, device=loss.device)
                burn_in_threshold = torch.tensor(burn_in_epoch - 1, device=loss.device)

                # 计算 upper_bound
                upper_bound = torch.where(epoch_tensor < burn_in_threshold, v_ones, v_zeros)

                # MentorNet 计算 v
                epoch_tensor = torch.full((batch, 1), fill_value=epoch, dtype=torch.float32).to(device) 
                input_data = torch.cat([loss.unsqueeze(1), lossdiff.unsqueeze(1), target.unsqueeze(1), epoch_tensor], dim=1).to('cuda')  # 拼接输入
                v = sigmoid(mentornet(input_data))  # MentorNet 计算权重 (batch, 1)
          
                v = torch.as_tensor(v, dtype=torch.float32).to(device)  # 转换为 float32 类型
                # v = torch.maximum(v, upper_bound)  # 限制 v 的最大值
                v = torch.maximum(v, upper_bound.to(v.device))
                

                # 1. 阻断 v 的梯度
                v = v.detach()  # v 的梯度被阻断，不会在反向传播中计算

                # 2. 加权损失
                weighted_loss_vector = loss * v  # 对每个样本的损失进行加权

                # 3. 计算加权损失的平均值
                loss = weighted_loss_vector.mean() # 返回加权损失的平均值作为最终损失

                # # --------- Mixup 数据增强 -------------
                # mixed_img, mixed_target = mixup_data(img, target, v)  # 使用 MentorNet 权重做 Mixup
                # feat_mix, bio_f_mix, clot_f_mix, score_mix, f_logits_mix, c_logits_mix, _, text_embeds_s_mix = model(
                #     mixed_img, instruction, label=mixed_target, cam_label=target_cam, view_label=target_view
                # )

                # loss_mixup = loss_fn(score_mix, f_logits_mix, c_logits_mix, feat_mix, bio_f_mix, clot_f_mix, mixed_target, text_embeds_s_mix, target_cam)

            # --------- 计算总损失并反向传播 -------------
            # loss_final = loss + loss_mixup
            scaler.scale(loss).backward()
            # print('loss:', loss)
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

            loss_meter.update(loss.item(), img.shape[0])
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

        # if epoch % checkpoint_period == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         if dist.get_rank() == 0:
        #             torch.save(model.state_dict(),
        #                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        #     else:
        #         torch.save(model.state_dict(),
        #                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch ==119:
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
                for n_iter, (img, instruction, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        batch = img.size(0)
                        # instruction = ('do_not_change_clothes',) * batch

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
    
    # 添加热力图生成相关配置
    # 在config.py中添加相应的配置项，默认不生成热力图
    generate_heatmaps = cfg.TEST.GENERATE_HEATMAPS if hasattr(cfg.TEST, 'GENERATE_HEATMAPS') else False
    heatmap_output_dir = os.path.join(cfg.OUTPUT_DIR, 'heatmaps') if generate_heatmaps else None
    
    # 只为一部分图像生成热力图，避免生成过多
    heatmap_count = 0
    heatmap_max = cfg.TEST.HEATMAP_MAX_IMAGES if hasattr(cfg.TEST, 'HEATMAP_MAX_IMAGES') else 20
    
    for n_iter, (img, instruction, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            batch = img.size(0)
            # instruction = ('do_not_change_clothes',) * batch
            feat, bio_f, clot_f, f_logits, c_logits, _, text_embeds_s = model(img, instruction, cam_label=camids, view_label=target_view )
            bio_clot_feat = torch.cat([bio_f, clot_f], dim=1)
            evaluator.update((bio_clot_feat, pid, camid))
            img_path_list.extend(imgpath)
            
            # 生成热力图（只为一部分样本生成）
            if generate_heatmaps and heatmap_count < heatmap_max:
                # 为当前批次生成热力图
                current_batch_size = min(batch, heatmap_max - heatmap_count)
                if current_batch_size > 0:
                    # 临时解除no_grad上下文以获取梯度
                    with torch.enable_grad():
                        # 为选定样本生成热力图
                        generate_attention_heatmap_matplotlib(
                            model, 
                            img[:current_batch_size].clone(), 
                            imgpath[:current_batch_size], 
                            heatmap_output_dir,
                            instruction[:current_batch_size],
                            device
                        )
                    heatmap_count += current_batch_size

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    if generate_heatmaps:
        logger.info(f"Attention heatmaps generated and saved to {heatmap_output_dir}")
        
    return cmc[0], cmc[4]

def generate_attention_heatmap_matplotlib(model, img_tensor, img_paths, output_dir, instructions=None, device='cuda'):
    """
    使用改进的方法为输入图像生成注意力热力图
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import os
    from scipy.ndimage import gaussian_filter
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 如果没有提供instructions，使用默认值
    batch_size = img_tensor.size(0)
    if instructions is None:
        instructions = ['do_not_change_clothes'] * batch_size
    
    # 为了获取梯度，需要启用梯度计算
    img_tensor.requires_grad_(True)
    
    # 获取模型输出
    camids = torch.zeros(batch_size, dtype=torch.long, device=img_tensor.device)
    target_view = torch.zeros(batch_size, dtype=torch.long, device=img_tensor.device)
    
    # 1. 分别计算生物特征和服装特征的注意力图
    feat, bio_f, clot_f, f_logits, c_logits, _, text_embeds_s = model(img_tensor, instructions, cam_label=camids, view_label=target_view)
    
    # 2. 使用组合特征作为目标（feat或bio_f+clot_f的组合）
    # 这里使用生物特征和服装特征的组合，可能更适合人物再识别任务
    target = torch.norm(torch.cat([bio_f, clot_f], dim=1), p=2, dim=1).mean()
    
    # 3. 反向传播
    model.zero_grad()
    target.backward()
    
    # 4. 获取输入图像的梯度
    gradients = img_tensor.grad.detach()
    
    # 5. 计算梯度的绝对值最大值（而不是平均值），增强对比度
    attention_maps = gradients.abs().max(dim=1)[0]  # 使用通道维度的最大值
    
    # 处理每个图像
    for i in range(batch_size):
        # 获取单个图像的注意力图
        attention_map = attention_maps[i].cpu().numpy()
        
        # 6. 应用高斯平滑减少噪声
        attention_map_smooth = gaussian_filter(attention_map, sigma=5)
        
        # 7. 应用阈值，只保留高激活区域
        threshold = np.percentile(attention_map_smooth, 70)  # 保留前30%的强度
        attention_map_filtered = np.where(attention_map_smooth > threshold, attention_map_smooth, 0)
        
        # 8. 归一化注意力图
        attention_map_filtered = (attention_map_filtered - attention_map_filtered.min()) / (attention_map_filtered.max() - attention_map_filtered.min() + 1e-8)
        
        # 读取原始图像
        img_path = img_paths[i]
        img_name = os.path.basename(img_path)
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        
        # 调整注意力图大小以匹配原始图像
        from skimage.transform import resize
        attention_map_resized = resize(attention_map_filtered, (orig_img.shape[0], orig_img.shape[1]), 
                                      preserve_range=True)
        
        # 创建图形
        plt.figure(figsize=(18, 6))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(orig_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # 热力图
        plt.subplot(1, 3, 2)
        plt.imshow(attention_map_resized, cmap='jet')
        plt.colorbar(label='Attention Intensity')
        plt.title('Attention Heatmap')
        plt.axis('off')
        
        # 叠加图
        plt.subplot(1, 3, 3)
        plt.imshow(orig_img)
        plt.imshow(attention_map_resized, cmap='jet', alpha=0.3)
        plt.colorbar(label='Attention Intensity')
        plt.title('Overlaid Heatmap')
        plt.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存结果
        output_path = os.path.join(output_dir, f"heatmap_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# def generate_attention_heatmap_matplotlib(model, img_tensor, img_paths, output_dir, instructions=None, device='cuda'):
#     """
#     使用matplotlib为输入图像生成注意力热力图
    
#     Args:
#         model: 训练好的模型
#         img_tensor: 输入图像张量
#         img_paths: 原始图像路径列表
#         output_dir: 热力图输出目录
#         instructions: 文本指令列表
#         device: 计算设备
#     """
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     import numpy as np
#     import os
#     from skimage.transform import resize
    
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 确保模型处于评估模式
#     model.eval()
    
#     # 如果没有提供instructions，使用默认值
#     batch_size = img_tensor.size(0)
#     if instructions is None:
#         instructions = ['do_not_change_clothes'] * batch_size
    
#     # 为了获取梯度，需要启用梯度计算
#     img_tensor.requires_grad_(True)
    
#     # 获取模型输出
#     camids = torch.zeros(batch_size, dtype=torch.long, device=img_tensor.device)
#     target_view = torch.zeros(batch_size, dtype=torch.long, device=img_tensor.device)
    
#     feat, bio_f, clot_f, f_logits, c_logits, _, text_embeds_s = model(img_tensor, instructions, cam_label=camids, view_label=target_view)
#     bio_clot_feat = torch.cat([bio_f, clot_f], dim=1)
    
#     # 使用特征的L2范数作为目标
#     target = torch.norm(bio_clot_feat, p=2, dim=1).mean()
    
#     # 反向传播
#     model.zero_grad()
#     target.backward()
    
#     # 获取输入图像的梯度
#     gradients = img_tensor.grad.detach()
    
#     # 计算梯度的绝对值平均，得到每个像素位置的重要性
#     attention_maps = gradients.abs().mean(dim=1)
    
#     # 处理每个图像
#     for i in range(batch_size):
#         # 获取单个图像的注意力图
#         attention_map = attention_maps[i].cpu().numpy()
        
#         # 归一化注意力图
#         attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
#         # 读取原始图像
#         img_path = img_paths[i]
#         img_name = os.path.basename(img_path)
#         orig_img = np.array(Image.open(img_path).convert('RGB'))
        
#         # 调整注意力图大小以匹配原始图像
#         attention_map_resized = resize(attention_map, (orig_img.shape[0], orig_img.shape[1]), 
#                                       preserve_range=True)
        
#         # 创建图形
#         plt.figure(figsize=(18, 6))
        
#         # 原始图像
#         plt.subplot(1, 3, 1)
#         plt.imshow(orig_img)
#         plt.title('Original Image')
#         plt.axis('off')
        
#         # 热力图
#         plt.subplot(1, 3, 2)
#         plt.imshow(attention_map_resized, cmap='jet')
#         plt.colorbar(label='Attention Intensity')
#         plt.title('Attention Heatmap')
#         plt.axis('off')
        
#         # 叠加图
#         plt.subplot(1, 3, 3)
#         plt.imshow(orig_img)
#         plt.imshow(attention_map_resized, cmap='jet', alpha=0.3)
#         plt.colorbar(label='Attention Intensity')
#         plt.title('Overlaid Heatmap')
#         plt.axis('off')
        
#         # 调整布局
#         plt.tight_layout()
        
#         # 保存结果
#         output_path = os.path.join(output_dir, f"heatmap_{os.path.splitext(img_name)[0]}.png")
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
#         # 保存PDF版本（矢量格式，适合论文）
#         output_path_pdf = os.path.join(output_dir, f"heatmap_{os.path.splitext(img_name)[0]}.pdf")
#         plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        
#         plt.close()
        
#         # # 单独保存热力图（便于论文中灵活使用）
#         # plt.figure(figsize=(6, 6))
#         # plt.imshow(attention_map_resized, cmap='jet')
#         # plt.colorbar(label='Attention Intensity')
#         # plt.axis('off')
#         # heatmap_only_path = os.path.join(output_dir, f"heatmap_only_{os.path.splitext(img_name)[0]}.png")
#         # plt.savefig(heatmap_only_path, dpi=300, bbox_inches='tight')
#         # plt.close()
