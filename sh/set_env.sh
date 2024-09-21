export PATH=:$PATH
prefix=/datapool/home/ph_teacher2/ls_experiment/images/pytorch_2.3.0-cuda11.8-cudnn8-devel
CUDA_HOME=/usr/local/cuda-11.8
CUDA_PATH=/usr/local/cuda-11.8

export PATH=$prefix$CUDA_HOME/bin:$PATH
export PATH=$prefix/opt/conda/bin:$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

# >>> conda initialize >>>
__conda_setup="$("$prefix/opt/conda/bin/conda" "shell.bash" "hook" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$prefix/opt/conda/etc/profile.d/conda.sh" ]; then
        . "$prefix/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="$prefix/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
