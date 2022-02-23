#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_LEVEL=INFO python3 $SCRIPT_DIR/../src/das/trainers/evaluate.py $@
