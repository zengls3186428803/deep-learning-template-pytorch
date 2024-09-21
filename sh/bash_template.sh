#!/bin/bash
echo "origin_paramters=<$@>"
ARGS=$(getopt --options="ab:c::" --longoptions="aa,bb:,cc::" -- "$@")
echo "after_getopt_parameters=<$ARGS>"
eval set -- "${ARGS}"
while true
do
        case "$1" in
                -a| --aa)
                        echo "option: $1"
                        shift
                        ;;
                -b| --bb)
                        echo "option $1=$2"
                        shift 2
                        ;;
                -c| --cc)
                        echo "option $1=$2"
                        shift 2
                        ;;
                --)
                        shift
                        break
                        ;;
                *)
                        echo "unrecognized option: $1"
                        break
                        ;;
        esac
done
echo "positional parameters=<$@>"

echo "$0 start======================================="

echo "$0 end========================================="