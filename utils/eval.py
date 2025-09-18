import numpy as np
import time
import datetime
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils


def mAP(scores_t2i, g_pids, q_pids, table=None):
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :10].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    if not table:
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP,
                   }

    return eval_result


@torch.no_grad()
def evaluation_itm(model, sims_matrix, image_embeds, text_embeds, text_atts, himage_embeds=None):
    model.eval()
    
    # Handle DistributedDataParallel model
    if hasattr(model, 'module'):
        model_module = model.module
    else:
        model_module = model

    header = 'Evaluation:'
    print('### Computing matching score -->')

    # 单卡直接处理全部数据
    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to('cuda')

    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=512, dim=0)

        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to('cuda')
        output = model_module.dual_attn(
            encoder_output, encoder_att,
            text_embeds[i].repeat(512, 1, 1),
            text_atts[i].repeat(512, 1)
        )[:, 0, :]
        score = model_module.itm_head(output)[:, 1]
        score_matrix_t2i[i, topk_idx] = score



    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    for i in range(sims_matrix.size(0)):
        score_matrix_t2i[i][score_matrix_t2i[i] == 1000.0] = replacement_tensor[i][score_matrix_t2i[i] == 1000.0]
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())


    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())

    if himage_embeds is None:
        score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i


    return score_matrix_t2i

@torch.no_grad()
def evaluation_itc(model, data_loader, tokenizer, device, config):
    model.eval()
    print('### Computing contrastive score -- >')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    text_embeds = []
    text_atts = []
    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        
        
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)
        text_feat = F.normalize(text_feat, dim=-1)

        text_embeds.append(text_embed)
        text_feats.append(text_feat)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    image_embeds = []
    image_feats = []
    himage_embeds = []
    himage_feats = []
    for image, himage, img_id in data_loader:
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        image_feat = model.get_image_feat(image_embed)
        image_feat = F.normalize(image_feat, dim=-1)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)

        if config.get('test_hflip', False):
            himage = himage.to(device)
            himage_embed, _ = model.get_vision_embeds(himage)
            himage_feat = model.get_image_feat(himage_embed)
            himage_feat = F.normalize(himage_feat, dim=-1)
            himage_embeds.append(himage_embed)
            himage_feats.append(himage_feat)


    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    if config.get('test_hflip', False):
        print('### itc_hflip')
        himage_embeds = torch.cat(himage_embeds, dim=0)
        himage_feats = torch.cat(himage_feats, dim=0)

        image_feats = (image_feats + himage_feats) / 2
    else:
        himage_embeds = None

    sims_matrix = image_feats @ text_feats.t()
    sims_matrix_t2i = sims_matrix.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Computing contrastive score time {}'.format(total_time_str))

    return sims_matrix_t2i, image_embeds, himage_embeds, text_embeds, text_atts
