set -e
echo "origin_paramters=<$@>"
ARGS=$(getopt --options="" --longoptions="" -- "$@")
echo "after_getopt_parameters=<$ARGS>"
eval set -- "${ARGS}"
while true
do
        case "$1" in
                --)
                        shift
                        break
                        ;;
                *)
                        echo "Unknown option: $1"
                        exit 1
                        ;;
        esac
done
echo "positional parameters=<$@>"

seed=$1
temperature=$2

function create_cuda_screen_sessions(){
        local cuda_indexes=$1
        local IFS_backup=$IFS
        IFS=","
        local cuda_index
        for cuda_index in $cuda_indexes; do
                local session_names=$(screen -list|grep -E -o "[0-9]*\.cuda_$cuda_index")
                if [[ -z $session_names ]];then
                        echo "new create session cuda_$cuda_index"
                        screen -dmS "cuda_${cuda_index}"
                fi
        done
        IFS=$IFS_backup
}

function dispatch_task_to_cuda(){
        local cuda_indexes=($(echo $1|sed "s/,/ /g"))
        shift 1
        tasks=($@)
        echo "tasks will be dispatch to ${cuda_indexes[@]}"
        local cuda_index
        len=${#cuda_indexes[@]}
        ((len=$len-1))
        IFS=$SPLIT
        for i in $(seq 0 $len); do
                session_name=$(screen -list|grep -E -o "[0-9]*\.cuda_${cuda_indexes[$i]}"|sed -n "1p")
                echo "use session $session_name"
                echo "screen -S $session_name -X stuff ${tasks[$i]}"
        done
}

