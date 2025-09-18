import logging
import os
import sys
import os.path as osp
import torch.distributed as dist

def setup_logger(name, save_dir, if_train, rank=None):
    """
    设置带分布式支持的日志记录器
    
    Args:
        name: logger名称
        save_dir: 保存目录
        if_train: 是否为训练模式
        rank: 分布式训练中的进程rank，如果None则自动获取
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 清除已有的handlers，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台输出 - 只让rank 0输出到控制台，避免重复
    if rank is None:
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0
    
    if rank == 0:  # 只有主进程输出到控制台
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 文件输出 - 每个进程都写入独立的日志文件
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        
        # 为每个进程创建独立的日志文件
        if dist.is_initialized():
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
        if dist.is_initialized():
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
        
        # 记录分布式训练信息
        if dist.is_initialized():
            logger.info(f"Distributed training setup - Rank: {rank}/{world_size-1}")
        else:
            logger.info("Single GPU training")

    return logger

def setup_distributed_logger(name, save_dir, if_train=True):
    """
    专门为分布式训练设计的日志记录器设置函数
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 创建主日志记录器
        logger = setup_logger(name, save_dir, if_train, rank)
        
        # 创建一个汇总日志文件（只有rank 0写入）
        if rank == 0 and save_dir:
            summary_log = os.path.join(save_dir, "training_summary.txt")
            with open(summary_log, 'w') as f:
                f.write(f"=== Distributed Training Summary ===\n")
                f.write(f"World Size: {world_size}\n")
                f.write(f"GPUs: {', '.join([f'GPU{i}' for i in range(world_size)])}\n")
                f.write(f"Start Time: {logger.handlers[-1].formatter.formatTime(logger.handlers[-1], logger.handlers[-1].__dict__.get('record', None))}\n")
                f.write("="*50 + "\n\n")
        
        return logger
    else:
        return setup_logger(name, save_dir, if_train)