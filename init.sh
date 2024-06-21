set -e
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1
hostname
nvcc --version
whereis conda
# apt update
# apt install vim wget axel curl -y

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
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
# conda init bash
echo "python of computaion node is: `python --version`"
# conda create -n lsvenv python=3.10 -y
# conda env list
conda activate lsvenv
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

cd /project/neural-ode
# wandb login xxxxxxxx
#python main.py
python start.py
#mkdir -p model/llama-7b
#huggingface-cli login --token xxxxx
#huggingface-cli download --repo-type model --local-dir model/llama-7b huggyllama/llama-7b
conda deactivate
