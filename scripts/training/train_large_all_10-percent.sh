#!/bin/bash

EXPERIMENT_NAME=large_all_10-percent

classy train generation data/compositional/all/10-percent \
    -n $EXPERIMENT_NAME \
    --profile large_all \
    --fp16 \
    --wandb dsrl-emnlp@$EXPERIMENT_NAME \
    -c \
        callbacks=evaluation \
        callbacks.0.settings.0.prediction_param_conf_path=configurations/prediction-params/beam.yaml \
        callbacks.0.settings.0.limit=100000 \
        callbacks.0.settings.0.token_batch_size=4096
