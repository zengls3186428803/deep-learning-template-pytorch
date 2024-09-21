# Deep learning Pytorch template

## package

- Configuration file management: hydra
- Distributed: torch.distributed, accelerate, deepspeed
- Visualization of the experiment: wandb
- Visusalization of the computing graph: torchviz

## Function calling convention

- DATASETMAPPING: dict mapping string to function that return (train_set, valid_set, test_set)
- MODELMAPPING: dict mapping string to function that return a model class

## accelerate example usage:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m accelerate.commands.launch \
--main_process_port $(shuf -i 10000-60000 -n 1) \
--config_file conf/accelerate/acc_ds_cfg.yaml \
example.py
```
