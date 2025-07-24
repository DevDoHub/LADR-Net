# import torch
# import numpy as np
# import os
# from utils.reranking import re_ranking


# def euclidean_distance(qf, gf):
#     m = qf.shape[0]
#     n = gf.shape[0]
#     dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist_mat.addmm_(1, -2, qf, gf.t())
#     return dist_mat.cpu().numpy()

# def cosine_similarity(qf, gf):
#     epsilon = 0.00001
#     dist_mat = qf.mm(gf.t())
#     qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
#     gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
#     qg_normdot = qf_norm.mm(gf_norm.t())

#     dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
#     dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
#     dist_mat = np.arccos(dist_mat)
#     return dist_mat


# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     # distmat g
#     #    q    1 3 2 4
#     #         4 1 2 3
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(-distmat, axis=1)
#     #  0 2 1 3
#     #  1 2 3 0
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]  # select one row
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
#         tmp_cmc = tmp_cmc / y
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)

#     return all_cmc, mAP


# class R1_mAP_eval():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
#         super(R1_mAP_eval, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking

#     def reset(self):
#         self.feats = []
#         self.text_embeds_s = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):  # called once for each batch
#         feat, text_embeds_s , pid, camid= output
#         self.feats.append(feat.cpu())
#         self.text_embeds_s.append(text_embeds_s.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):  # called after each epoch
#         feats = torch.cat(self.feats, dim=0)
#         text_embeds_s = torch.cat(self.text_embeds_s, dim=0)
        
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
#             text_embeds_s = torch.nn.functional.normalize(text_embeds_s, dim=1, p=2)  # normalize text embeddings
        
#         # query - 文本嵌入
#         q_text_embeds_s = text_embeds_s[:self.num_query]   
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
        
#         # gallery - 图像特征
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])

#         if self.reranking:
#             print('=> Enter reranking for T2I retrieval')
#             distmat = re_ranking(q_text_embeds_s, gf, k1=20, k2=6, lambda_value=0.3)
#         else:
#             print('=> Computing DistMat with cosine similarity for T2I retrieval')
#             # 对于T2I检索，使用余弦相似度更合适
#             distmat = self.text_to_image_distance(q_text_embeds_s, gf)
            
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

#         return cmc, mAP, distmat, self.pids, self.camids, q_text_embeds_s, gf

#     def text_to_image_distance(self, text_embeds, image_feats):
#         """
#         计算文本嵌入和图像特征之间的余弦相似度
#         """
#         # 确保特征被正确归一化
#         text_norm = torch.nn.functional.normalize(text_embeds, dim=1, p=2)
#         image_norm = torch.nn.functional.normalize(image_feats, dim=1, p=2)
        
#         # 计算余弦相似度
#         similarity = torch.mm(text_norm, image_norm.t())
        
        
#         return similarity.cpu().numpy()



import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


