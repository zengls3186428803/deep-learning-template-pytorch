# deep-learning-template-pytorch

## description for packages and software

* python
    * hydra for configuration and multirun
    * wandb for logging information
    * tqdm for progress bar
    * torchviz for visualizing computing graph

* shell(bash,zsh)
    * slurm for High-performance computing
    * singularity for container

## description for files and directories

**singularity.sh** for start a container

**init.sh** is executed once enter the singularity container

**class_*** is python class for *

**conf** contains configuration yaml for hydra

**get_dataloader** contains function of get dataloader

## run

```
python main.py
```