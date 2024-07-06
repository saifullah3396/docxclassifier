#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../src:$SCRIPT_DIR/../external/torchfusion/src:$PYTHONPATH

TYPE=standard
POSITIONAL_ARGS=()

usage() {
  echo "Usage:"
  echo "./train.sh --type=<type>"
  echo ""
  echo " --type : Command to run. "
  echo " -h | --help : Displays the help"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    shift # past argument
    usage
    exit
    ;;
  -t | --type)
    TYPE="$2"
    shift # past argument
    shift # past value
    ;;
  -c | --cfg_root)
    CFG_ROOT="$2"
    shift # past argument
    shift # past value
    ;;
  *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift                   # past argument
    ;;
  esac
done

# you can set LOG_LEVEL=DEBUG to see more logs
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
python3 -W ignore $SCRIPT_DIR/../src/docxclassifier/runners/train.py --config-path ../../../cfg/ --config-name hydra +train=hydra "${@:1}"
