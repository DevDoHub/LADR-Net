# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


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
        def loss_func(score, f_logits, c_logits, feat, bio_f, clot_f, target, text_embeds_s, target_cam):
            import torch
            num_class = score.size(0)
            LOSS = 0 
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':#是否平滑标签
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
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target.long(), reduction='none')
                        LOSS += cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                        # LOSS[:, 0] = ID_LOSS #TODO

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet_loss(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet_loss(feat, target, text_embeds_s)[0]
                            LOSS += cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                            # LOSS[:, 1] = TRI_LOSS #TODO

                if 'bio' in cfg.MODEL.FUSION_BRANCH :
                    if isinstance(f_logits, list):
                        BIO_ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        BIO_ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        BIO_ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        BIO_ID_LOSS = F.cross_entropy(f_logits, target.long(), reduction='none')
                        LOSS += cfg.MODEL.BIO_ID_LOSS_WEIGHT * BIO_ID_LOSS
                        # LOSS[:, 2] = BIO_ID_LOSS #TODO


                    if isinstance(bio_f, list):
                            BIO_TRI_LOSS = [triplet_loss(feats, target)[0] for feats in feat[1:]]
                            BIO_TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            BIO_TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            BIO_TRI_LOSS = triplet_loss(bio_f, target, text_embeds_s)[0] 
                            LOSS += cfg.MODEL.BIO_TRIPLET_LOSS_WEIGHT * BIO_TRI_LOSS
                            # LOSS[:, 3] = BIO_TRI_LOSS #TODO   

                if 'clot' in cfg.MODEL.FUSION_BRANCH :
                    if isinstance(f_logits, list):
                        CLOT_ID_LOSS = [F.cross_entropy(c_logits, target) for scor in score[1:]]
                        CLOT_ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        CLOT_ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        CLOT_ID_LOSS = F.cross_entropy(c_logits, target.long(), reduction='none')
                        LOSS += cfg.MODEL.CLOT_ID_LOSS_WEIGHT * CLOT_ID_LOSS
                        # LOSS[:, 4] = CLOT_ID_LOSS #TODO 

                    if isinstance(clot_f, list):
                            CLOT_TRI_LOSS = [triplet_loss(feats, target)[0] for feats in feat[1:]]
                            CLOT_TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            CLOT_TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            CLOT_TRI_LOSS = triplet_loss(clot_f, target, text_embeds_s)[0] 
                            LOSS += cfg.MODEL.CLOT_TRIPLET_LOSS_WEIGHT * CLOT_TRI_LOSS
                            # LOSS[:, 5] = CLOT_TRI_LOSS #TODO 

                return LOSS
            # return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                    #            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion