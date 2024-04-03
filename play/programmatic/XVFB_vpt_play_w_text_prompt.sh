#!/bin/bash

RUN_NAME=$1

CONFIG_PATH="minedreamer/play/config/programmatic/${RUN_NAME}.yaml"

xvfb-run /opt/conda/envs/minerl/bin/python minedreamer/play/programmatic/vpt_play_w_text_prompt.py \
--config ${CONFIG_PATH}