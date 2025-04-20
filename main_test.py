import os
import subprocess

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 训练命令
command = [
    "python", "test.py",
    "--config_file", "configs/market/swin_base.yml",
    "TEST.WEIGHT", "'./log/market/swin_base/transformer_120.pth'",
    "MODEL.SEMANTIC_WEIGHT", "0.2"
]

# 执行训练命令
subprocess.run(command)
