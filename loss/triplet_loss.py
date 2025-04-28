import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


# class TripletLoss(object):
#     """
#     Triplet loss using HARDER example mining,
#     modified based on original triplet loss using hard example mining
#     """

#     def __init__(self, margin=None, hard_factor=0.0):
#         self.margin = margin
#         self.hard_factor = hard_factor
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()

#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an = hard_example_mining(dist_mat, labels)

#         #  dist_ap *= (1.0 + self.hard_factor)
#         #  dist_an *= (1.0 - self.hard_factor)

#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an
def cosine_simalirity(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine
def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p1 = sorted_mat_distance[:, 0]
    hard_p_indice1 = positive_indices[:, 0]
    
    hard_p2 = sorted_mat_distance[:, 1]
    hard_p_indice2 = positive_indices[:, 1]
    
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    # import pdb;pdb.set_trace()
    if (indice):
        return hard_p1, hard_p2, hard_n, hard_p_indice1, hard_p_indice2, hard_n_indice
    return hard_p1, hard_p2, hard_n
class TripletLoss(nn.Module):

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin, reduction='none').cuda()

    def forward(self, emb, label, clot_feats_s):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)
        
        mat_dist_clot_feats_s = cosine_simalirity(clot_feats_s, clot_feats_s)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        
        dist_ap1, dist_ap2, dist_an, dist_ap1_indice, dist_ap2_indice, dist_an_indice = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap1.size(0)
        
        alpha1 = torch.rand(dist_ap1_indice.shape).to(dist_ap1_indice.device)
        for b_index1, index1 in enumerate(dist_ap1_indice):
            alpha1[b_index1] = mat_dist_clot_feats_s[b_index1][index1].detach()
        
        alpha2 = torch.rand(dist_ap2_indice.shape).to(dist_ap2_indice.device)
        for b_index2, index2 in enumerate(dist_ap2_indice):
            alpha2[b_index2] = mat_dist_clot_feats_s[b_index2][index2].detach()
        
        alphan = torch.rand(dist_an_indice.shape).to(dist_an_indice.device)
        for b_indexn, indexn in enumerate(dist_an_indice):
            alphan[b_indexn] = mat_dist_clot_feats_s[b_indexn][indexn].detach()
        
        y11 = torch.ones_like(dist_ap1)
        y11_m = torch.ones_like(dist_ap1)
        y11[alpha1 < alpha2] = -1
        y11_m[alpha1 == alpha2] = 0
        
        loss11 = self.margin_loss(dist_ap2*y11_m, dist_ap1*y11_m + self.margin*(alpha1 - alpha2 - y11), y11)#TODO
        # loss11 = torch.clamp(-y13 * (dist_ap1 - dist_an) + self.margin, min=0)

        y13 = torch.ones_like(dist_ap1)
        
        dist_ap1 =  dist_ap1 + self.margin*(alpha1 - 1)
        
        loss13 = self.margin_loss(dist_an, dist_ap1, y13)
        
        y23 = torch.ones_like(dist_ap2)
        
        dist_ap2 =  dist_ap2 + self.margin*(alpha2 - 1)
        
        loss23 = self.margin_loss(dist_an, dist_ap2, y23)
        loss = 0.1 * loss11 + loss13
        prec = (dist_an.data > dist_ap1.data).sum() * 1. / y11.size(0)
        return loss, prec