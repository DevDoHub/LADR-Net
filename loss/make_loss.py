# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .smi_loss import compute_sdm
from .contrastive_loss import get_contrastive_loss
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
        def loss_func(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, text_score, target_cam, epoch):
            LOSS = 0
            # if epoch < 40:
            #     loss_weight = 1
            #     smi_weight = 0
            #     alignment_loss_weight = 0
            # elif epoch < 80:
            #     smi_weight = 0.1
            #     alignment_loss_weight = 0.05
            #     loss_weight = 1
            # elif epoch < 160:
            #     smi_weight = 0.1 + 0.9 * (epoch - 80) / 80.0  # 0.1~1线性增长
            #     alignment_loss_weight = 0.05 + 0.95 * (epoch - 80) / 80.0  # 0.05~1线性增长
            #     loss_weight = 1.0 - 0.7 * (epoch - 80) / 80.0  # 1~0.3线性下降（如需同步变化）
            # else:
            #     smi_weight = 1.0
            #     alignment_loss_weight = 1.0
            #     loss_weight = 0.3
            # if epoch < 40:
            #     loss_weight = 1
            #     feat_loss_weight = 1
            #     smi_weight = 0
            #     alignment_loss_weight = 0
            # elif epoch < 80:
            #     smi_weight = 1
            #     alignment_loss_weight = 0.1 + 0.9 * (epoch - 40) / 40.0 #从 0.1 线性增长到 1.0（随 epoch 从40到80）
            #     loss_weight = 1.0 - 0.5 * (epoch - 80) / 80.0  # 从 1.0 线性下降到 0.5
            #     feat_loss_weight = 1.0 - 0.9 * (epoch - 40) / 40.0 #从 1.0 线性下降到 0.1（随 epoch 从40到80）
            # elif epoch < 120:
            #     smi_weight = 1
            #     alignment_loss_weight = 1
            #     loss_weight = 0.5 * (1.0 - 0.9 * (epoch - 40) / 40.0) 
            #     feat_loss_weight = 0.02
            # else:
            #     smi_weight = 1.0
            #     alignment_loss_weight = 1.5
            #     loss_weight = 0.01
            #     feat_loss_weight = 0
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet_loss(feat, target, text_embeds_s)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    # img_cls cross_entropy
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                        LOSS += 0.3 * ID_LOSS

                    #text_cls cross_entropy
                    if isinstance(text_score, list):
                        ID_LOSS = [F.cross_entropy(text_score, target) for scor in text_score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(text_score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(text_score, target)
                        LOSS += 0.3 * ID_LOSS

                    # loss_itc = get_contrastive_loss(feat, text_embeds_s, idx = target)
                    # LOSS += loss_itc
                    
                    smi_loss = compute_sdm(feat, text_embeds_s, target, 50)  
                    # if epoch > 20:
                    LOSS += 1.5 * smi_loss

                    # alignment_loss = dual_attn_alignment_loss(
                    #     bio_f=bio_f,
                    #     clot_f=clot_f,
                    #     img_cls=feat,
                    #     text_cls=text_embeds_s,
                    #     cos_weight=10.0,
                    #     euclidean_weight=1.0
                    # )
                    # LOSS += alignment_loss_weight * alignment_loss

                    # if isinstance(feat, list):
                    #     TRI_LOSS = [triplet_loss(feats, target)[0] for feats in feat[1:]]
                    #     TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    #     TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    # else:
                    #     TRI_LOSS = triplet_loss(feat, target, text_embeds_s)[0]
                    #     LOSS += 5 * TRI_LOSS

                # if 'bio' in cfg.MODEL.FUSION_BRANCH :
                #     if isinstance(f_logits, list):
                #         BIO_ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                #         BIO_ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                #         BIO_ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                #     else:
                #         BIO_ID_LOSS = F.cross_entropy(f_logits, target)
                #         LOSS += loss_weight * cfg.MODEL.BIO_ID_LOSS_WEIGHT * BIO_ID_LOSS


                #     if isinstance(bio_f, list):
                #         BIO_TRI_LOSS = [triplet_loss(feats, target)[0] for feats in feat[1:]]
                #         BIO_TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                #         BIO_TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                #     else:
                #         BIO_TRI_LOSS = triplet_loss(bio_f, target, clot_f)[0] 
                #         LOSS += loss_weight * cfg.MODEL.BIO_TRIPLET_LOSS_WEIGHT * BIO_TRI_LOSS   

                # if 'clot' in cfg.MODEL.FUSION_BRANCH :
                #     if isinstance(f_logits, list):
                #         CLOT_ID_LOSS = [F.cross_entropy(c_logits, target) for scor in score[1:]]
                #         CLOT_ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                #         CLOT_ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                #     else:
                #         CLOT_ID_LOSS = F.cross_entropy(c_logits, target)
                #         LOSS +=  loss_weight * cfg.MODEL.BIO_TRIPLET_LOSS_WEIGHT * CLOT_ID_LOSS


                return LOSS, smi_loss
            # return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                    #            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion



def dual_attn_alignment_loss(
    bio_f, clot_f, img_cls, text_cls,
    cos_weight=1.0, euclidean_weight=0.1
):
    """
    bio_f, clot_f: cross-attention 特征
    img_cls, text_cls: 原始 CLS 特征
    让 bio_f 和 clot_f 固定，不参与梯度，原始特征向它靠拢
    """
    # -----------------------
    # 固定 bio_f 和 clot_f
    # -----------------------
    bio_fixed = bio_f.detach()
    clot_fixed = clot_f.detach()

    # -----------------------
    # Cosine loss
    # -----------------------
    bio_norm = F.normalize(bio_fixed, dim=-1)
    img_norm = F.normalize(img_cls, dim=-1)
    clot_norm = F.normalize(clot_fixed, dim=-1)
    text_norm = F.normalize(text_cls, dim=-1)

    loss_bio_cos = 1 - F.cosine_similarity(bio_norm, img_norm, dim=-1).mean()
    loss_clot_cos = 1 - F.cosine_similarity(clot_norm, text_norm, dim=-1).mean()

    # -----------------------
    # Euclidean loss
    # -----------------------
    loss_bio_euc = F.mse_loss(img_cls, bio_fixed)  # 注意顺序
    loss_clot_euc = F.mse_loss(text_cls, clot_fixed)

    # -----------------------
    # 总损失
    # -----------------------
    total_loss = cos_weight * (loss_bio_cos + loss_clot_cos) + \
                 euclidean_weight * (loss_bio_euc + loss_clot_euc)

    return total_loss
