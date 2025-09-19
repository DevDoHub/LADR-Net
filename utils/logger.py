import logging
import os
import sys
import os.path as osp
import torch.distributed as dist

def setup_logger(name, save_dir, if_train, rank=None, distributed=None):
    """
    设置日志记录器，支持单GPU和分布式训练
    
    Args:
        name: logger名称
        save_dir: 保存目录
        if_train: 是否为训练模式
        rank: 分布式训练中的进程rank，如果None则自动获取
        distributed: 是否强制使用分布式模式，None则自动检测
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 清除已有的handlers，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 自动检测分布式环境
    is_distributed = distributed if distributed is not None else dist.is_initialized()
    
    # 获取rank信息
    if rank is None:
        try:
            rank = dist.get_rank() if is_distributed else 0
        except:
            rank = 0
    
    # 控制台输出 - 只让rank 0输出到控制台，避免重复
    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 文件输出
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        
        # 根据是否分布式训练决定文件名
        if is_distributed:
            world_size = dist.get_world_size()
            if if_train:
                log_filename = f"train_log_rank{rank}_of_{world_size}.txt"
            else:
                log_filename = f"test_log_rank{rank}_of_{world_size}.txt"
        else:
            if if_train:
                log_filename = "train_log.txt"
            else:
                log_filename = "test_log.txt"
        
        fh = logging.FileHandler(os.path.join(save_dir, log_filename), mode='w')
        fh.setLevel(logging.DEBUG)
        
        # 为分布式训练添加rank信息到日志格式
        if is_distributed:
            formatter = logging.Formatter(
                f"[Rank {rank}] %(asctime)s %(name)s %(levelname)s: %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s: %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 记录训练模式信息
        if is_distributed:
            world_size = dist.get_world_size()
            logger.info(f"Distributed training setup - Rank: {rank}/{world_size-1}")
        else:
            logger.info("Single GPU training")

    return logger

def setup_distributed_logger(name, save_dir, if_train=True):
    """
    专门为分布式训练设计的日志记录器设置函数
    向后兼容性函数
    """
    return setup_logger(name, save_dir, if_train)

# 为了保持向后兼容，提供原始接口
def setup_simple_logger(name, save_dir, if_train):
    """
    简单日志记录器设置（原logger.py的接口）
    向后兼容性函数
    """
    return setup_logger(name, save_dir, if_train, distributed=False)