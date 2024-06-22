import functools
import os
import pickle
from functools import wraps
from datetime import datetime, timedelta
import time
import wandb


def cache_to_disk(root_datadir):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_filename = root_datadir + "/" + f"{func_name}.pkl"
            print("cache_filename=", cache_filename)

            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    print(f"Loading cached data for {func.__name__}")
                    return pickle.load(f)

            result = func(*args, **kwargs)
            print("caching " + cache_filename)
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
                print(f"Cached data for {func.__name__}")
            return result

        return wrapper

    return decorator


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
