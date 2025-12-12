from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    ## ==>> 参数加载部分
    # 参数解析
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 配置加载
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # 配置冻结
    cfg.freeze()
    # 设置随机种子
    set_seed(cfg.SOLVER.SEED)

    ## ==>> 日志处理部分
    # 日志记录器设置
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)

    ## 分布式训练设置
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 加载数据集
    train_loader, train_loader_normal,  query_loader, gallery_loader , num_test, num_classes = make_dataloader(cfg)
    logger.info('using {} images for training'.format(len(train_loader.dataset)))

    # 创建模型
    logger.info("Creating model: {}".format(cfg.MODEL.NAME))
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    # model.load_param('/root/SOLIDER-REID-PRO/transformer_30.pth')
    ## 记录一些信息
    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total model parameters: {}".format(total_params))
    # 使用 named_parameters() 获取模型的参数名称和参数本身
    with open("model_parameters.txt", "w") as f:
        for name, param in model.named_parameters():
            # 将参数的名称写入 txt 文件
            f.write(name + "\n")

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        query_loader, 
        gallery_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_test, args.local_rank
    )
