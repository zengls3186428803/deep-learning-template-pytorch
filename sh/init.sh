set -e
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1
hostname
nvcc --version
whereis conda


# >>> conda initialize >>>
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
if conda info --root &>/dev/null; then
    echo "Conda is already initialized."
else
    echo "Conda is not initialized. Initializing Conda for Bash."
    conda init bash
    source ~/.bashrc
fi
echo "python of computaion node is: `python --version`"

ENV_NAME="loraga"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Conda environment '$ENV_NAME' does not exist. Creating a new environment."
    conda create --name "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
fi
echo "python of conda is: `python --version`"
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install -r /project/loraga/requirements.txt
# pip install -U huggingface_hub
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# pip install torchdiffeq
# pip install hydra-core
# pip install wandb
# pip install tqdm

cd /project
python test_cuda.py

cd /project/GGAM
python main_entry.py
conda deactivate
