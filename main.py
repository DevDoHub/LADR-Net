import os
import subprocess

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 训练命令
command = [
    "python", "train.py",
    "--config_file", "configs/msmt17/swin_base.yml",
    "MODEL.PRETRAIN_CHOICE", "'self'",
    "MODEL.PRETRAIN_PATH", "'./checkpoint_tea.pth'",
    "OUTPUT_DIR", "'./log/msmt17/swin_base'",
    "SOLVER.BASE_LR", "0.0002",
    "SOLVER.OPTIMIZER_NAME", "'SGD'",
    "MODEL.SEMANTIC_WEIGHT", "0.2"
]

# 执行训练命令
subprocess.run(command)
