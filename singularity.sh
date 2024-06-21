#!/bin/bash
#SBATCH --partition=phys_hq
#SBATCH --job-name=zengls_ode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --output=ode_Multirun_output.txt
#SBATCH --chdir=.

ARGS=$(getopt --options= --longoptions="init-path::" -- "$@")
echo $ARGS
eval set -- "${ARGS}"
while true
do
        case "$1" in
                --init-path)
                        initpath=$2
                        echo "option: $1"
                        echo "value: $2"
                        echo ""
                        shift 2
                        ;;
                --)
                        shift
                        break
                        ;;
                *)
                        echo "unrecognized option"
                        break
        esac
done
echo "rest option(s): $@"

echo "$0 start======================================="
hostname
nvidia-smi
container=pytorch_2.3.0-cuda11.8-cudnn8-devel
SFLAG="--writable --nv"
init_path=${initpath-"/project/ode_init.sh"}
echo "$container start*******************************"
singularity exec $SFLAG $container bash $init_path
echo "$container end*********************************"

echo "$0 end========================================="
