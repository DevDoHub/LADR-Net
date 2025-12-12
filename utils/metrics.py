from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import torch.nn as nn
from utils.eval import evaluation_itm

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    # 强化兼容性：accept list/tuple/np.ndarray/torch.Tensor for q_pids/g_pids
    # normalize similarity -> torch tensor
    if isinstance(similarity, torch.Tensor):
        sim = similarity.clone().detach()
    else:
        sim = torch.tensor(np.asarray(similarity))

    # 获得排序/topk 索引
    if get_mAP:
        indices = torch.argsort(sim, dim=1, descending=True)
    else:
        _, indices = torch.topk(sim, k=max_rank, dim=1, largest=True, sorted=True)

    # 将 q_pids/g_pids 转为 1D CPU long tensor
    def _to_1d_long(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().view(-1)
        else:
            arr = np.asarray(x)
            t = torch.from_numpy(arr).view(-1)
        return t.long()

    q_pids_t = _to_1d_long(q_pids)
    g_pids_t = _to_1d_long(g_pids)

    # 通过 indices 从 gallery pid 中取标签（在 CPU 上）
    idx_cpu = indices.cpu()
    try:
        pred_labels = g_pids_t[idx_cpu]
    except Exception:
        # 兜底：使用 numpy 索引再转为 tensor
        pred_labels = torch.from_numpy(np.array(g_pids_t.numpy())[idx_cpu.numpy()]).long()

    # matches: q x k (bool)
    matches = pred_labels.eq(q_pids_t.view(-1, 1))

    # CMC
    topk = max_rank
    all_cmc = matches[:, :topk].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    # mINP: 对每个 query，找最后一个正确匹配的位置，计算对应的精度；无匹配的 query 跳过
    tmp_cmc = matches.cumsum(1)
    inp_list = []
    for i, match_row in enumerate(matches):
        pos = match_row.nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        last = pos[-1].item()
        inp_val = tmp_cmc[i, last] / (last + 1.0)
        inp_list.append(inp_val)
    if len(inp_list) == 0:
        mINP = torch.tensor(0.0)
    else:
        mINP = torch.stack(inp_list).mean() * 100

    # AP / mAP：对每个 query，用 matches 与累积精度计算 AP；当 num_rel==0 时 AP=0
    num_rel = matches.sum(1)  # q
    denom = num_rel.clone()
    denom[denom == 0] = 1  # 避免除以0

    tmp_cmc_local = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc_stack = torch.stack(tmp_cmc_local, 1) * matches
    AP = tmp_cmc_stack.sum(1) / denom
    AP[num_rel == 0] = 0.0
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, query_loader, gallery_loader):
        self.gallery_loader = gallery_loader # gallery
        self.query_loader = query_loader # query
        self.logger = logging.getLogger("transreid.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        
        # Handle DistributedDataParallel model
        if hasattr(model, 'module'):
            model_module = model.module
        else:
            model_module = model

        qids, gids, qfeats, gfeats, qembed, gembed,text_attr= [], [], [], [], [], [], []
        # text
        for img, caption_tokens, pid, _, _, _, img_path in self.query_loader:
            # caption = caption.to(device)
            img = img.to(device)
            with torch.no_grad():
                global_feat, local_feat_all, text_feat = model_module( img, caption_tokens, label=pid
            )
                # text_embeds = text_outputs.last_hidden_state 
                # text_embeds = text_embeds @ model.text_projection  # 将 (batch, seq_len, 768) 转为 (batch, seq_len, 1024)
                # text_feat = text_embeds[:, 0, :]
            qids.append(pid) # flatten 
            qfeats.append(global_feat)
            qembed.append(local_feat_all)
            text_attr.append(text_feat)
                # image
        for img, caption_tokens, pid, _, _, _, img_path in self.gallery_loader:
            img = img.to(device)
            with torch.no_grad():
                global_feat, local_feat_all, text_feat = model_module( img, caption_tokens, label=pid
            )

            gids.append(pid) # flatten 
            gfeats.append(global_feat)
            gembed.append(local_feat_all)
        # text_atts = torch.cat(text_attr, 0)
        # qids = torch.cat(qids, 0)
        # qfeats = torch.cat(qfeats, 0)
        # qembed = torch.cat(qembed, 0)
        # 把收集到的 list 展开并转为 tensor / 合并到一个张量里
        def _flatten_ids(id_list):
            vals = []
            for x in id_list:
                if x is None:
                    continue
                if torch.is_tensor(x):
                    arr = x.detach().cpu().numpy().ravel().tolist()
                    vals.extend(arr)
                elif isinstance(x, (list, tuple, np.ndarray)):
                    vals.extend(list(x))
                else:
                    vals.append(x)
            if len(vals) == 0:
                return torch.tensor([], dtype=torch.long)
            return torch.tensor(vals, dtype=torch.long)

        def _cat_feats(feat_list):
            if len(feat_list) == 0:
                return torch.tensor([])
            # 如果元素已经是 tensor，则直接 cat
            if torch.is_tensor(feat_list[0]):
                return torch.cat(feat_list, 0)
            else:
                return torch.tensor(np.vstack(feat_list))

        try:
            text_atts = _cat_feats(text_attr) if len(text_attr) > 0 else torch.tensor([])
        except Exception:
            text_atts = torch.tensor([])

        qids = _flatten_ids(qids)
        gids = _flatten_ids(gids)
        qfeats = _cat_feats(qfeats)
        gfeats = _cat_feats(gfeats)
        qembed = _cat_feats(qembed) if len(qembed) > 0 else torch.tensor([])
        gembed = _cat_feats(gembed) if len(gembed) > 0 else torch.tensor([])


        # gids = torch.cat(gids, 0)
        # gfeats = torch.cat(gfeats, 0)
        # gembed = torch.cat(gembed, 0)
        # 确保返回的都是 tensor / 合并好的结构
        # gids, gfeats 等已经在上面合并处理

        return qfeats, gfeats, qids, gids, qembed, gembed, text_atts
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids, qembed, gembed, text_atts= self._compute_embedding(model)
        
        # Handle DistributedDataParallel model for evaluation_itm
        if hasattr(model, 'module'):
            model_for_eval = model.module
        else:
            model_for_eval = model

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()
        # score_matrix_t2i = evaluation_itm(
        #     model_for_eval, similarity, gembed, qembed, text_atts
        # )
        
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        # 将返回值规范为 numpy 数组 / 标量，兼容 gallery 数量 < 10 的情况
        if torch.is_tensor(t2i_cmc):
            t2i_cmc_arr = t2i_cmc.cpu().numpy()
        else:
            t2i_cmc_arr = np.asarray(t2i_cmc)
        t2i_mAP_val = float(t2i_mAP.detach().cpu().item()) if torch.is_tensor(t2i_mAP) else float(t2i_mAP)
        t2i_mINP_val = float(t2i_mINP.detach().cpu().item()) if torch.is_tensor(t2i_mINP) else float(t2i_mINP)

        # 安全索引，若索引超出范围则取最后一个可用值或 0
        def _safe_rank(cmc_arr, idx):
            if cmc_arr.size == 0:
                return 0.0
            if idx < cmc_arr.shape[0]:
                return float(cmc_arr[idx])
            return float(cmc_arr[-1])

        r1 = _safe_rank(t2i_cmc_arr, 0)
        r5 = _safe_rank(t2i_cmc_arr, 4)
        r10 = _safe_rank(t2i_cmc_arr, 9)

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', r1, r5, r10, t2i_mAP_val, t2i_mINP_val])

        # if i2t_metric:
        #     i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
        #     i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
        #     table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # # table.float_format = '.4'
        # table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        # table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        # table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        # table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        # table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        # self.logger.info('\n' + str(table))
        
        return r1, r5, r10, t2i_mAP_val, t2i_mINP_val
