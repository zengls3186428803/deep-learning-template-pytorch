# Deep learning Pytorch template

## Usage

```bash
git clone https://github.com/zengls3186428803/deep-learning-template-pytorch.git
cd deep-learning-template-pytorch
git submodule update
pip install -r requirements.txt
```

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
