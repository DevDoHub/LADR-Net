# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

def attention_loss(attn_weights, target_attn):
    """
    使用 KL 散度或均方误差计算注意力损失。
    """
    target_attn = target_attn / target_attn.sum(dim=-1, keepdim=True)  # 归一化目标注意力
    loss = F.kl_div(attn_weights.log(), target_attn, reduction='batchmean')  # KL 散度
    return loss

def attention_regularization_loss(attn_weights):
    """
    使用负熵正则化，让注意力更加集中。
    """
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # 注意力分布的熵
    return entropy.mean()  # 平均熵作为损失

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            # triplet = TripletLoss()
            triplet_loss = TripletLoss(margin=0.3).cuda()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, target_cam, attn_weights=None, target_attn=None):
            LOSS = 0

            # 1. 分类损失（ID Loss）
            if isinstance(score, list):
                ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
            else:
                ID_LOSS = F.cross_entropy(score, target)
            LOSS += cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS

            # 2. 全局特征三元组损失
            if isinstance(feat, list):
                TRI_LOSS = [triplet_loss(feats, target, text_embeds_s)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet_loss(feat[0], target, text_embeds_s)[0]
            else:
                TRI_LOSS = triplet_loss(feat, target, text_embeds_s)[0]
            LOSS += cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # 3. BIO 分支的损失
            if 'bio' in cfg.MODEL.FUSION_BRANCH:
                # BIO 分类损失
                BIO_ID_LOSS = F.cross_entropy(f_logits, target)
                LOSS += cfg.MODEL.BIO_ID_LOSS_WEIGHT * BIO_ID_LOSS

                # BIO 三元组损失（注意避免重复计算）
                BIO_TRI_LOSS = triplet_loss(bio_f, target, text_embeds_s)[0]
                LOSS += cfg.MODEL.BIO_TRIPLET_LOSS_WEIGHT * BIO_TRI_LOSS

            # 4. CLOT 分支的损失
            if 'clot' in cfg.MODEL.FUSION_BRANCH:
                # CLOT 分类损失
                CLOT_ID_LOSS = F.cross_entropy(c_logits, target)
                LOSS += cfg.MODEL.CLOT_ID_LOSS_WEIGHT * CLOT_ID_LOSS

                # CLOT 三元组损失（避免重复计算）
                CLOT_TRI_LOSS = triplet_loss(clot_f, target, text_embeds_s)[0]
                LOSS += cfg.MODEL.CLOT_TRIPLET_LOSS_WEIGHT * CLOT_TRI_LOSS

            # if attn_weights is not None and target_attn is not None:
            #     ATTENTION_LOSS = attention_loss(attn_weights, target_attn)
            #     LOSS += 1 * ATTENTION_LOSS
            #     # LOSS += cfg.MODEL.ATTENTION_LOSS_WEIGHT * ATTENTION_LOSS

            if attn_weights is not None:
                ATTENTION_REG_LOSS = attention_regularization_loss(attn_weights)
                LOSS += 1 * ATTENTION_REG_LOSS
                

            return LOSS

            #     return LOSS
            # # return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
            #         #            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            # else:
            #     print('expected METRIC_LOSS_TYPE should be triplet'
            #           'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


