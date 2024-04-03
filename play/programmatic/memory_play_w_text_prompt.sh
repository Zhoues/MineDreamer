#!/bin/bash

RUN_NAME=$1

CONFIG_PATH="minedreamer/play/config/programmatic/${RUN_NAME}.yaml"
MEMORY_DIR="data/memory"

python minedreamer/play/programmatic/memory_play_w_text_prompt.py \
--config ${CONFIG_PATH} \
--memory_dir ${MEMORY_DIR}