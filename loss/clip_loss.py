import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        # init_logit_scale = np.log(1 / 0.07)  # ≈ 2.66
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    def cosine_simalirity(self, x, y):  # ✅ 添加self参数
        bs1, bs2 = x.size(0), y.size(0)
        frac_up = torch.matmul(x, y.transpose(0, 1))
        frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                    (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
        cosine = frac_up / frac_down
        return cosine

    def forward(
            self,
            image_features,
            text_features,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image = self.cosine_simalirity(
            image_features,
            text_features
        )
        logits_per_text = logits_per_image.t()

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss

