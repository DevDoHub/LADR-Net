#!/usr/bin/env python3
"""
数据并行训练启动脚本
将数据分布到2个GPU上进行并行训练
"""

import subprocess
import sys
import os
import time
import signal
import torch

torch.version.__version__


def signal_handler(sig, frame):
    print('\n训练被用户中断，正在清理...')
    sys.exit(0)

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    # 基础打印
    print("="*60)
    print("启动数据并行训练...")
    print("配置信息:")
    print(f"  - PyTorch版本: {torch.version.__version__}")  # 1.7.1+cu110
    gpu_count = torch.cuda.device_count()
    # gpu_count = 3  # 手动设置
    print(f"可用GPU数量: {gpu_count}") # 2个4090
    # for i in range(gpu_count):
    #     gpu_name = torch.cuda.get_device_name(i)
    #     print(f"GPU {i} 型号: {gpu_name}")
    print("  - 训练方式: 数据并行 (DistributedDataParallel)")
    print("  - 后端: NCCL")
    print("  - 启动方式: torch.distributed.launch (适配PyTorch 1.7.1)")
    print("  - 每个GPU处理不同的数据批次，模型参数在GPU间同步")
    print("="*60)

    cmd = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={gpu_count}',  # 每个节点的进程数（GPU数量）
        '--nnodes=1',          # 节点数量
        '--node_rank=0',       # 当前节点的rank
        '--master_addr=localhost',  # 主节点地址
        '--master_port=12355',      # 主节点端口
        'train.py',
        '--config_file', 'configs/cuhkpedes/swin_base_data_parallel.yml',
        'MODEL.PRETRAIN_CHOICE', "'self'",
        'MODEL.PRETRAIN_PATH', "'./checkpoint_tea.pth'",
        'OUTPUT_DIR', "'./log/cuhkpedes/swin_base_data_parallel'",
        'MODEL.SEMANTIC_WEIGHT', '0.2',
        'SOLVER.BASE_LR', '0.00001',
        'SOLVER.IMS_PER_BATCH', '64',
        'TEST.IMS_PER_BATCH', '32',
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print("-"*60)
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        # env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(gpu_count)])
        env['CUDA_VISIBLE_DEVICES'] = '0,1'  # 手动设置
        
        # 创建输出目录
        output_dir = './log/sda/swin_base_data_parallel'
        os.makedirs(output_dir, exist_ok=True)
        
        # 将启动信息写入日志
        with open(os.path.join(output_dir, 'training_launch.log'), 'w') as f:
            f.write(f"Training launched at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("Environment variables:\n")
            for key, value in env.items():
                if 'CUDA' in key or 'MASTER' in key:
                    f.write(f"  {key}={value}\n")

        print("开始训练...")
        result = subprocess.run(cmd, check=True, env=env)

        print("="*60)
        print("🎉 数据并行训练成功完成!")
        print(f"📁 日志目录: {output_dir}")
        print("📊 检查以下文件获取详细信息:")
        print("   - train_log_rank0_of_2.txt (主进程日志)")
        print("   - train_log_rank1_of_2.txt (辅进程日志)")
        print("   - training_summary.txt (训练汇总)")
        print("="*60)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print("❌ 训练失败!")
        print(f"错误代码: {e.returncode}")
        print("请检查错误日志以获取详细信息")
        return 1
    except KeyboardInterrupt:
        print("⚠️  训练被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 发生未预期的错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())