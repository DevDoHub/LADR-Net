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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go


# 读取Pickle文件
def read_pickle_file(file_path):
    """
    读取Pickle文件
    
    Args:
        file_path: Pickle文件路径
    
    Returns:
        data: 读取的数据
    """
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 保存Pickle文件
def save_pickle_file(data, file_path):
    """
    保存数据到Pickle文件
    
    Args:
        data: 要保存的数据
        file_path: Pickle文件路径
    """
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# 可视化t-SNE
def visualize_tsne(features, labels, output_dir, perplexity=30, n_iter=1000, filename='tsne_visualization'):
    """
    使用t-SNE可视化特征空间
    
    Args:
        features: 特征向量 numpy数组，形状为 [n_samples, n_features]
        labels: 标签 numpy数组，形状为 [n_samples]
        output_dir: 输出目录
        perplexity: t-SNE的perplexity参数
        n_iter: t-SNE的迭代次数
        filename: 输出文件名
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始进行t-SNE降维, 处理{len(features)}个样本...")
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_embedded = tsne.fit_transform(features)
    
    print("t-SNE降维完成，开始绘图...")
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 为每个类别绘制散点图
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            features_embedded[indices, 0],
            features_embedded[indices, 1],
            c=[colors[i]],
            label=f'ID {label}',
            alpha=0.7,
            s=40
        )
    
    # 如果类别太多，不显示图例
    if len(unique_labels) <= 20:  # 限制图例条目数量
        plt.legend(loc='best')
        
    plt.title('t-SNE Visualization of ReID Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE可视化已保存至 {output_path}")
    
    # 保存PDF版本用于论文
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"t-SNE可视化(PDF)已保存至 {pdf_path}")
    
    plt.close()

# 改进的UMAP可视化 - 支持2D和3D
def visualize_umap_enhanced(features, labels, output_dir, n_neighbors=30, min_dist=0.0, 
                           n_components=2, filename='umap_visualization', max_display_ids=50):
    """
    使用UMAP可视化特征空间 - 增强版本
    
    Args:
        features: 特征向量 numpy数组，形状为 [n_samples, n_features]
        labels: 标签 numpy数组，形状为 [n_samples]
        output_dir: 输出目录
        n_neighbors: UMAP的n_neighbors参数，增加以获得更全局的结构
        min_dist: UMAP的min_dist参数，减少以获得更紧密的聚类
        n_components: 降维后的维数 (2或3)
        filename: 输出文件名
        max_display_ids: 最大显示的ID数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始进行UMAP降维, 处理{len(features)}个样本...")
    unique_labels = np.unique(labels)
    print(f"总共有{len(unique_labels)}个不同的ID")
    
    # 使用UMAP降维
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                       n_components=n_components, random_state=42, 
                       metric='cosine')  # 使用cosine距离，更适合特征向量
    features_embedded = reducer.fit_transform(features)
    
    print(f"UMAP降维完成({n_components}D)，开始绘图...")
    
    if n_components == 2:
        # 2D可视化 - 只显示部分ID以避免颜色混乱
        _visualize_2d_selective(features_embedded, labels, output_dir, filename, max_display_ids)
        # 2D可视化 - 密度图
        _visualize_2d_density(features_embedded, labels, output_dir, filename)
    elif n_components == 3:
        # 3D可视化
        _visualize_3d_interactive(features_embedded, labels, output_dir, filename, max_display_ids)
        _visualize_3d_matplotlib(features_embedded, labels, output_dir, filename, max_display_ids)

def _visualize_2d_selective(features_embedded, labels, output_dir, filename, max_display_ids):
    """2D可视化 - 选择性显示部分ID"""
    unique_labels = np.unique(labels)
    
    # 随机选择一部分ID进行可视化
    if len(unique_labels) > max_display_ids:
        selected_labels = np.random.choice(unique_labels, max_display_ids, replace=False)
        print(f"随机选择{max_display_ids}个ID进行可视化")
    else:
        selected_labels = unique_labels
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 为选中的ID绘制散点图
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_labels)))
    for i, label in enumerate(selected_labels):
        indices = labels == label
        plt.scatter(
            features_embedded[indices, 0],
            features_embedded[indices, 1],
            c=[colors[i]],
            label=f'ID {label}',
            alpha=0.8,
            s=30
        )
    
    # 为其他ID绘制灰色背景点
    other_indices = ~np.isin(labels, selected_labels)
    if np.any(other_indices):
        plt.scatter(
            features_embedded[other_indices, 0],
            features_embedded[other_indices, 1],
            c='lightgray',
            alpha=0.3,
            s=10,
            label='Other IDs'
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title(f'UMAP Visualization of ReID Features (显示{len(selected_labels)}个ID)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{filename}_2d_selective.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D选择性UMAP可视化已保存至 {output_path}")
    plt.close()

def _visualize_2d_density(features_embedded, labels, output_dir, filename):
    """2D可视化 - 密度图"""
    plt.figure(figsize=(12, 10))
    
    # 绘制密度散点图
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], 
               c=labels, cmap='tab20', alpha=0.6, s=20)
    
    plt.colorbar(label='Person ID')
    plt.title('UMAP Visualization - Density Plot')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{filename}_2d_density.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D密度UMAP可视化已保存至 {output_path}")
    plt.close()

def _visualize_3d_matplotlib(features_embedded, labels, output_dir, filename, max_display_ids):
    """3D可视化 - matplotlib版本"""
    unique_labels = np.unique(labels)
    
    # 选择部分ID进行可视化
    if len(unique_labels) > max_display_ids:
        selected_labels = np.random.choice(unique_labels, max_display_ids, replace=False)
    else:
        selected_labels = unique_labels
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为选中的ID绘制散点图
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_labels)))
    for i, label in enumerate(selected_labels):
        indices = labels == label
        ax.scatter(
            features_embedded[indices, 0],
            features_embedded[indices, 1],
            features_embedded[indices, 2],
            c=[colors[i]],
            label=f'ID {label}',
            alpha=0.8,
            s=30
        )
    
    # 为其他ID绘制灰色背景点
    other_indices = ~np.isin(labels, selected_labels)
    if np.any(other_indices):
        ax.scatter(
            features_embedded[other_indices, 0],
            features_embedded[other_indices, 1],
            features_embedded[other_indices, 2],
            c='lightgray',
            alpha=0.3,
            s=10,
            label='Other IDs'
        )
    
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    ax.set_title(f'3D UMAP Visualization of ReID Features (显示{len(selected_labels)}个ID)')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{filename}_3d_matplotlib.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D matplotlib UMAP可视化已保存至 {output_path}")
    plt.close()

def _visualize_3d_interactive(features_embedded, labels, output_dir, filename, max_display_ids):
    """3D可视化 - 交互式版本（需要安装plotly）"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        unique_labels = np.unique(labels)
        
        # 选择部分ID进行可视化
        if len(unique_labels) > max_display_ids:
            selected_indices = np.isin(labels, np.random.choice(unique_labels, max_display_ids, replace=False))
        else:
            selected_indices = np.ones(len(labels), dtype=bool)
        
        # 创建交互式3D散点图
        fig = px.scatter_3d(
            x=features_embedded[selected_indices, 0],
            y=features_embedded[selected_indices, 1],
            z=features_embedded[selected_indices, 2],
            color=labels[selected_indices].astype(str),
            title='Interactive 3D UMAP Visualization of ReID Features',
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3'},
            opacity=0.7
        )
        
        fig.update_traces(marker_size=3)
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            width=1200,
            height=800
        )
        
        # 保存交互式HTML文件
        output_path = os.path.join(output_dir, f'{filename}_3d_interactive.html')
        fig.write_html(output_path)
        print(f"3D交互式UMAP可视化已保存至 {output_path}")
        
    except ImportError:
        print("未安装plotly，跳过交互式3D可视化")

# 保留原有的UMAP函数以保持兼容性
def visualize_umap(features, labels, output_dir, n_neighbors=15, min_dist=0.1, n_components=2, filename='umap_visualization'):
    """原有的UMAP可视化函数 - 保持兼容性"""
    visualize_umap_enhanced(features, labels, output_dir, n_neighbors, min_dist, n_components, filename)


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
    
    # 用于收集特征和标签进行可视化
    all_features = []
    all_pids = []
    
    for n_iter, (img, instruction, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            batch = img.size(0)

            feat, bio_f, clot_f, f_logits, c_logits, _, text_embeds_s = model(img, instruction, cam_label=camids, view_label=target_view)
            bio_clot_feat = torch.cat([bio_f, clot_f], dim=1)
            evaluator.update((bio_clot_feat, pid, camid))
            img_path_list.extend(imgpath)
            
            # 收集特征和标签用于可视化
            all_features.append(bio_clot_feat.cpu().numpy())
            
            # 处理pid的不同数据类型
            if isinstance(pid, torch.Tensor):
                all_pids.extend(pid.numpy())
            elif isinstance(pid, (list, tuple)):
                all_pids.extend(pid)
            else:
                all_pids.append(pid)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    # 进行增强的UMAP可视化
    logger.info("开始生成增强UMAP可视化...")
    all_features = np.concatenate(all_features, axis=0)
    all_pids = np.array(all_pids)
    
    # 创建可视化输出目录
    vis_output_dir = os.path.join(cfg.OUTPUT_DIR, 'visualizations')
    
    # 生成2D UMAP可视化
    logger.info("生成2D UMAP可视化...")
    visualize_umap_enhanced(
        features=all_features,
        labels=all_pids,
        output_dir=vis_output_dir,
        n_neighbors=30,  # 增加邻居数以获得更全局的结构
        min_dist=0.0,    # 减少最小距离以获得更紧密的聚类
        n_components=2,
        filename='inference_umap_2d',
        max_display_ids=50
    )
    
    # 生成3D UMAP可视化
    logger.info("生成3D UMAP可视化...")
    visualize_umap_enhanced(
        features=all_features,
        labels=all_pids,
        output_dir=vis_output_dir,
        n_neighbors=30,
        min_dist=0.0,
        n_components=3,
        filename='inference_umap_3d',
        max_display_ids=50
    )
    
    # 也保存特征和标签以备后续分析
    features_save_path = os.path.join(vis_output_dir, 'inference_features.pkl')
    save_pickle_file({
        'features': all_features,
        'pids': all_pids,
        'img_paths': img_path_list
    }, features_save_path)
    logger.info(f"特征数据已保存至 {features_save_path}")
    
    return cmc[0], cmc[4]