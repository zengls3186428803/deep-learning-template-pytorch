# deep-learning-template-pytorch

## preparation

* hydra for configuration
* wandb for logging information
* tqdm for progress bar

## directory structure

* conf(configuration in yaml)
    * config.yaml(main configuration file)
    * algorithm/(configuration of scheduler and optimizer)
    * model/(configuration of model)
    * data/(configuration of dataset and dataloader)
    * train/(configuration of training)
* config_class(configuration class correspond to conf/)
    * algorithmConfig.py
    * dataConfig.py
    * modelConfig.py
    * trainConfig.py
    * wandbConfig.py
* utils
    * seed_all.py(set seed for reproduction of experiment)
    * decorator(counting time, logging information )
* trainer.py(encapsulate the training process)
