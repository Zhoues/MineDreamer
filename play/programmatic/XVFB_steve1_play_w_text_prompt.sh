#!/bin/bash

RUN_NAME=$1

CONFIG_PATH="minedreamer/play/config/programmatic/${RUN_NAME}.yaml"
PRIOR_WEIGHTS_PATH="data/weights/steve1/steve1_prior.pt"

xvfb-run /opt/conda/envs/minerl/bin/python minedreamer/play/programmatic/steve1_play_w_text_prompt.py \
--config ${CONFIG_PATH} \
--prior_weights_path ${PRIOR_WEIGHTS_PATH}