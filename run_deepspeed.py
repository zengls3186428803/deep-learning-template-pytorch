import deepspeed
from deepspeed.launcher.runner import main

# 定义DeepSpeed的命令行参数
deepspeed_args = [
    "--num_gpus", "1",  # 指定使用的GPU数量
    "try_ds.py",  # 训练脚本的路径
]

# 调用DeepSpeed的main函数启动训练
main(deepspeed_args)
