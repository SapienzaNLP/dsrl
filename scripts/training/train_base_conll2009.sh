#!/bin/bash

EXPERIMENT_NAME=base_conll2009

classy train generation data/compositional/conll2009 \
    -n $EXPERIMENT_NAME \
    --profile base_conll2009 \
    --fp16 \
    --wandb dsrl-emnlp@$EXPERIMENT_NAME \
    -c \
        callbacks=evaluation \
        callbacks.0.settings.0.prediction_param_conf_path=configurations/prediction-params/beam.yaml \
        callbacks.0.settings.0.limit=100000 \
        callbacks.0.settings.0.token_batch_size=4096
