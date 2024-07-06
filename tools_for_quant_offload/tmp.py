import accelerate
import deepspeed
import torch
from accelerate import DeepSpeedPlugin
from torch.utils.data import Dataset


def get_offload_model_using_deep_speed(
        model: torch.nn.Module,
        dataset: Dataset = None,
        offload_proportion: float = 0.5,
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
            "stage3_param_persistence_threshold": int(
                (1 - offload_proportion) * sum(p.numel() for p in model.parameters())),
            "stage3_max_live_parameters": int((1 - offload_proportion) * sum(p.numel() for p in model.parameters())),
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
    }
    ds_config.update(config)
    plugin = DeepSpeedPlugin(
        model,
    )
    plugin.deepspeed_config.update(
        ds_config
    )
    accelerator_tmp = accelerate.Accelerator(deepspeed_plugin=plugin)
    model = accelerator_tmp.prepare(model)
    return model


def get_offload_model_using_deep_speed(
        model: torch.nn.Module,
        dataset: Dataset = None,
        offload_proportion: float = 0.5,
        offload_strategy: str = None,
        config=None,
        world_size: int = 1,
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
                # "pin_memory": True
            },
            "reduce_bucket_size": 5000,
            "stage3_prefetch_bucket_size": 5000,
            "stage3_param_persistence_threshold": 1000,
            "stage3_max_live_parameters": 1e7,
            "stage3_max_reuse_distance": 1e5,
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

    total_num_paras = len(list(model.parameters()))
    num_paras_resident_in_gpu = int((1 - offload_proportion) * total_num_paras)
    ds_config.update(
        {"stage3_max_live_parameters": num_paras_resident_in_gpu // world_size},
    )
    ds_config.update(
        {"stage3_param_persistence_threshold": num_paras_resident_in_gpu}
    )
    print("(num_paras_in_gpu, total_num_paras)=", num_paras_resident_in_gpu, total_num_paras)
    ds_model, _optimizer, _dataloader, _scheduler = deepspeed.initialize(
        model=model,
        training_data=dataset,
        config=ds_config,
    )
    return ds_model


def get_dispatch_model(
        model: torch.nn.Module,
        strategy="only_cpu",
        device_map=None,
        max_memory=None,
        offload_dir=None,
) -> torch.nn.Module:
    """
    This function doesn't seem to work
    """
    import accelerate
    from accelerate.big_modeling import dispatch_model
    from accelerate.big_modeling import cpu_offload
    match strategy:
        case "only_cpu":
            model = cpu_offload(model)
        case "auto":
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
            )
            print("auto device map is ", device_map)
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                state_dict=model.state_dict(),
            )
        case "device_map":
            assert device_map is not None, "device map is None"
            model = dispatch_model(model, device_map=device_map)
        case _:
            print("No offload policy is specified, so do nothing")
    return model
