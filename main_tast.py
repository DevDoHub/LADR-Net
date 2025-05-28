import os
import subprocess

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 训练命令
command = [
    "python", "test.py",
    "--config_file", "configs/market/swin_base.yml",
    "MODEL.PRETRAIN_CHOICE", "'self'",
    # "TEST.WEIGHT", "/root/SOLIDER-REID-PRO/log/cuhk03/swin_base/transformer_120.pth",
    "TEST.WEIGHT", "/root/SOLIDER-REID-PRO/log/market/swin_base/transformer_120.pth",
    "OUTPUT_DIR", "'./log/market/swin_base'",
    "SOLVER.BASE_LR", "0.0002",
    "SOLVER.OPTIMIZER_NAME", "'SGD'",
    "MODEL.SEMANTIC_WEIGHT", "0.2"
]

# 执行训练命令
subprocess.run(command)
