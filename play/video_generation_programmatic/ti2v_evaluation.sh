#!/bin/bash

RUN_NAME=$1
MODEL_NAME=$2

CONFIG_PATH="minedreamer/play/config/programmatic/${RUN_NAME}.yaml"

/opt/conda/envs/minerl/bin/python minedreamer/play/video_generation_programmatic/ti2v_evaluation.py \
--config ${CONFIG_PATH} \
--model_name ${MODEL_NAME}