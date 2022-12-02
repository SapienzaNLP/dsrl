#!/bin/bash

# -s: system prediction path [conll] format
# -g: gold path [conll] format
# --train_dataset_path: path to training set. Required for MFS computation
# --conll_2009_predicate_match: whether or not to apply conll_2009 predicate match criterion (sense number only)

python scripts/performance_analysis/performance_analyzer.py \
        -s data/CoNLL2009_dev.pred.txt \
        -g data_share/conll2009/original/CoNLL2009_dev.txt \
        --train_dataset_path data_share/conll2009/original/CoNLL2009_train.txt \
        --conll2009_predicate_match