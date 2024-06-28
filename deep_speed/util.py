import torch
from torch.utils.data import Dataset
import deepspeed


def get_offload_model_using_deep_speed(
        model: torch.nn.Module,
        dataset: Dataset = None,
        offload_proportion: float = 1,
        offload_strategy: str = None,
        config=None,
):
    if config is None:
        config = dict()
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "reduce_bucket_size": 1e5,
            "stage3_prefetch_bucket_size": 1e5,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e5,
            "stage3_max_reuse_distance": 0,
        },
        "train_batch_size": 5,
        "train_micro_batch_size_per_gpu": 5,
        "wall_clock_breakdown": False,
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
        "logging": {
            "level": "info"
        },
        "profiling": {
            "enabled": True,
            "profile_step": 1,
            "module_name": "deepspeed",
            "report_name": "ds_report.html"
        }
    }
    ds_config.update(config)

    # total_num_paras = len(list(model.parameters()))
    total_num_paras = sum(p.numel() for p in model.parameters())
    num_paras_resident_in_gpu = int((1 - offload_proportion) * total_num_paras)


    ds_config["zero_optimization"].update(
        {
            "stage3_param_persistence_threshold": num_paras_resident_in_gpu,
            "stage3_max_live_parameters": num_paras_resident_in_gpu,
        }
    )
    print("deepspeed config is_is_is", ds_config)
    print(f"deep_speed_config {num_paras_resident_in_gpu} in gpu, total is {total_num_paras} ")
    ds_model, _optimizer, _dataloader, _scheduler = deepspeed.initialize(
        model=model,
        # model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    return ds_model
