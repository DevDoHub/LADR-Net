
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_contrastive_loss(self, image_feat, text_feat, idx):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        image_feat_all = image_feat
        text_feat_all = text_feat
     
        logits = image_feat_all @ text_feat_all.t() / self.temp

        idx = idx.view(-1, 1)
        assert idx.size(0) == image_feat.size(0)
        idx_all = idx
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)

        # if 0 < self.epsilon < 1:
        #     _, num_classes = logits.shape
        #     labels = (1 - self.epsilon) * labels + self.epsilon / num_classes

        loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2