import torch
import wandb
from functools import wraps


def wandb_loger(desc: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # 将结果记录到 wandb
            print("decorator")
            return result

        return wrapper

    return decorator


class CovTrainer:
    @wandb_loger(desc="")
    def get_eigenvalue(self, matrix: torch.Tensor, flag=""):
        return {
            flag + "eigenvalue": torch.linalg.eigvalsh(matrix)
        }


# 示例用法
trainer = CovTrainer()
x = trainer.get_eigenvalue(torch.Tensor([[1, 2], [2, 1]]), flag="uuu")
print(x)
