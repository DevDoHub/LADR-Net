import torch
import torch.nn as nn
from torch.nn import functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from torch.nn import init
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a

# from model.backbones.tokenization_bert import BertTokenizer
from model.backbones.xbert import BertConfig, BertForMaskedLM
from transformers import BertModel
from model.backbones.crossattention import CrossBlock
from model.backbones.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 1024
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 1024
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(1024, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 1024)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False

        # self.encoder, base_cfg = build_CLIP_from_openai_pretrained('ViT-B/16', (384, 128), 16)
        bert_config = BertConfig.from_json_file('./config/config_bert.json')
        # self.text_embed_dim = 512
        self.text_encoder = BertForMaskedLM.from_pretrained(
        "./bert-base-uncased1/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594", 
        local_files_only=True, config=bert_config
        )
        self.text_embed_dim = 768  # bert-large 的隐藏层维度是1024

    
        # # 替换原有的 nn.Sequential
        # self.fusion_layers_T2I = nn.ModuleList([
        #     CrossBlock(dim=self.text_embed_dim, num_heads=16, qkv_bias=True),
        #     # CrossBlock(dim=1024, num_heads=16, qkv_bias=True),
        #     # CrossBlock(dim=1024, num_heads=16, qkv_bias=True)
        # ])
        # self.fusion_layers_I2T = nn.ModuleList([
        #     CrossBlock(dim=1024, num_heads=16, qkv_bias=True),
        #     CrossBlock(dim=1024, num_heads=16, qkv_bias=True),
        #     CrossBlock(dim=1024, num_heads=16, qkv_bias=True)
        # ])

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, pretrained=model_path, convert_weights=convert_weights, semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            # self.classifier_TEXT = nn.Linear(self.text_embed_dim, self.num_classes, bias=False)
            # self.classifier_TEXT.apply(weights_init_classifier)
            # self.text_classifier_projector = nn.Linear(self.text_embed_dim, self.in_planes, bias=False)
            # self.text_classifier_projector.apply(weights_init_xavier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.num_features = self.in_planes

        self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)

        self.feat_bn = nn.BatchNorm1d(self.in_planes)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # self.avgpool_image = nn.AdaptiveAvgPool1d(3)
        # self.avgpool_text = nn.AdaptiveAvgPool1d(3)
        # self.vision_proj = build_itc_mlp(3072, 1024, 0.5)
        # self.text_proj = build_itc_mlp(3072, 1024, 0.5)

        # self.image_projection = nn.Parameter(torch.empty(1024, 768))
        # nn.init.normal_(self.image_projection, std=0.01)  # 初始化投影矩阵

        
        # self.temp = nn.Parameter(torch.ones([]) * 0.07)
        # self.itm_head = build_itm_mlp(input_dim=self.text_embed_dim, output_dim=2)

    # def get_image_feat(self, image_embeds):

    #     x = self.avgpool_image(image_embeds.transpose(1, 2))
    #     x = x.transpose(1, 2)
    #     x = torch.cat([x[:, 0, :],   x[:, 1, :], x[:, 2, :]], dim=1)
    #     image_feat = self.vision_proj(x)

    #     return image_feat


    # def get_text_feat(self, text_embeds):

    #     x = self.avgpool_text(text_embeds[:, 1:, ].transpose(1, 2))  # B C 3
    #     x = x.transpose(1, 2)  # B 3 C
    #     x = torch.cat([x[:, 0, :], x[:, 0, :], x[:, 1, :], x[:, 1, :], x[:, 2, :], x[:, 2, :]], dim=1)
    #     text_feat = self.text_proj(x)

        # return text_feat
    


    # def dual_attn(self, image_embeds, image_atts, text_embeds, text_atts):
    #     encoder = self.text_encoder.bert
    #     return encoder(encoder_embeds=text_embeds,
    #             attention_mask=text_atts,
    #             encoder_hidden_states=image_embeds,
    #             encoder_attention_mask=image_atts,
    #             return_dict=True,
    #             mode='fusion',
    #             ).last_hidden_state
    
    # def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx):
    #     """
    #     Matching Loss with hard negatives
    #     """
    #     bs = image_embeds.size(0)

    #     image_feat = F.normalize(image_feat, dim=-1)
    #     text_feat = F.normalize(text_feat, dim=-1)


    #     with torch.no_grad():
    #         sim_i2t = image_feat @ text_feat.t() / self.temp
    #         sim_t2i = text_feat @ image_feat.t() / self.temp
    #         weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
    #         weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

    #         idx = idx.view(-1, 1)
    #         assert idx.size(0) == bs
    #         mask = torch.eq(idx, idx.t())
    #         weights_i2t.masked_fill_(mask, 0)
    #         weights_t2i.masked_fill_(mask, 0)

    #     image_embeds_neg = []
    #     image_atts_neg = []
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_t2i[b], 1).item()
    #         image_embeds_neg.append(image_embeds[neg_idx])
    #         image_atts_neg.append(image_atts[neg_idx])
    #     image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    #     image_atts_neg = torch.stack(image_atts_neg, dim=0)

    #     text_embeds_neg = []
    #     text_atts_neg = []
    #     for b in range(bs):
    #         neg_idx = torch.multinomial(weights_i2t[b], 1).item()
    #         text_embeds_neg.append(text_embeds[neg_idx])
    #         text_atts_neg.append(text_atts[neg_idx])
    #     text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    #     text_atts_neg = torch.stack(text_atts_neg, dim=0)

    #     text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
    #     text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
    #     image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
    #     image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

    #     cross_pos = self.dual_attn(image_embeds, image_atts, text_embeds,
    #                                       text_atts)[:, 0, :]
    #     cross_neg = self.dual_attn(image_embeds_all, image_atts_all, text_embeds_all,
    #                                       text_atts_all)[:, 0, :]


    #     output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
    #     itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
    #                             torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
    #     itm_loss = F.cross_entropy(output, itm_labels)

    #     return itm_loss

    # def get_contrastive_loss(self, image_feat, text_feat, idx):

    #     image_feat = F.normalize(image_feat, dim=-1)
    #     text_feat = F.normalize(text_feat, dim=-1)

    #     image_feat_all = image_feat
    #     text_feat_all = text_feat
     
    #     logits = image_feat_all @ text_feat_all.t() / self.temp

    #     idx = idx.view(-1, 1)
    #     assert idx.size(0) == image_feat.size(0)
    #     idx_all = idx
    #     pos_idx = torch.eq(idx_all, idx_all.t()).float()
    #     labels = pos_idx / pos_idx.sum(1, keepdim=True)

    #     loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
    #     loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

    #     return (loss_i2t + loss_t2i) / 2

    def forward(self, x, instruction, label=None, cam_label= None, view_label=None):
        # 获取模型当前所在的设备
        device = next(self.parameters()).device
        
        text_outputs = self.text_encoder.bert(
            input_ids=instruction['input_ids'].to(device),
            token_type_ids=instruction['token_type_ids'].to(device),
            attention_mask=instruction['attention_mask'].to(device),
            return_dict=True, mode='text'
        )
        text_embeds = text_outputs.last_hidden_state 
        # text_embeds = text_embeds @ self.text_projection  # 将 (batch, seq_len, 768) 转为 (batch, seq_len, 1024)
        text_feat = text_embeds[:, 0, :]

        global_feat, featmaps = self.base(x, text_feat=text_feat)
        # 在 forward 或 __init__ 中添加
        # total_params = count_parameters(self.text_encoder)
        # print(f"Base model parameters: {total_params:,}")#global_feat全局特征 featmaps[-1]最后阶段输出的([64, 1024, 12, 4])
        batch = featmaps[-1].size(0)
        local_feat_all = featmaps[-1].view(batch, 1024, 12 * 4).permute(0, 2, 1)
        image_embeds = torch.cat((global_feat.unsqueeze(1), local_feat_all), dim=1)#TODO
        # image_embeds = image_embeds @ self.image_projection
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        # image_feat = image_embeds[:, 0, :]

        # if self.reduce_feat_dim:
        #     global_feat = self.fcneck(global_feat)

        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)
        # image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)

        # loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts=instruction['attention_mask'].to('cuda'), text_feat=text_feat, idx=label)

        # loss_itc = self.get_contrastive_loss(image_feat, text_feat, label)


        # if self.reduce_feat_dim:
        #     logits = self.fcneck(local_feat_all)

        # feat = self.bottleneck(image_feat)
        # text_feat = self.feat_bn(text_feat)
        # feat_cls = self.dropout(feat)
        # text_cls = self.dropout(text_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
                # text_score = self.classifier(text_feat, label)
            else:
                cls_score = self.classifier(feat_cls)
                # text_score = self.classifier_TEXT(text_feat)
                # text_score = self.classifier(self.text_classifier_projector(text_feat))
            return global_feat, cls_score, local_feat_all, text_feat# global feature for triplet loss  
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat, featmaps
            else:
                # print("Test with feature before BN")
                return global_feat, local_feat_all, text_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
def build_itc_mlp(input_dim, output_dim, dropout_p=0):
    mlp = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(input_dim, output_dim),
    )
    init.normal_(mlp[2].weight.data, std=0.00001)
    init.constant_(mlp[2].bias.data, 0.0)
    return mlp

def build_itm_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)