from functools import wraps
from datetime import datetime, timedelta
import time
import wandb


def timer(data_format="ms"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            begin_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            cost = (end_time - begin_time).seconds
            print(func.__name__ + "运行了" + f" {cost // 60} min {cost % 60}s", )
            return result

        return wrapper

    return decorator


def wandb_loger(desc: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # for k, v in result.items():
            #     result[k] = desc + str(v)
            wandb.log(result)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    @timer()
    def f():
        time.sleep(2)


    f()
