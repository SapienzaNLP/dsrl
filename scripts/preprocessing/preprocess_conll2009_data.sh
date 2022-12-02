#!/bin/bash

FRAMES_PATH=data_share/conll2009/frames/conll2009_frames.json
ARGM_DEFINITIONS_PATH=data_share/conll2009/frames/conll2009_argm_definitions.json

INPUT_PATHS=(\
    data_share/conll2009/original/CoNLL2009_train.txt \
    data_share/conll2009/original/CoNLL2009_train_10-percent.txt \
    data_share/conll2009/original/CoNLL2009_train_25-percent.txt \
    data_share/conll2009/original/CoNLL2009_train_50-percent.txt \
    data_share/conll2009/original/CoNLL2009_train_75-percent.txt \
    data_share/conll2009/original/CoNLL2009_dev.txt \
    data_share/conll2009/original/CoNLL2009_test.txt \
    data_share/conll2009/original/CoNLL2009_test-ood.txt \
    data_share/conll2009/original/CoNLL2009_sample.200.txt \
    data_share/conll2009/original/CoNLL2009_sample.10k.txt \
)

OUTPUT_PATHS=(\
    data_share/conll2009/jsonl/CoNLL2009_train.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_train_10-percent.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_train_25-percent.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_train_50-percent.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_train_75-percent.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_dev.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_test.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_test-ood.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_sample.200.conll2009 \
    data_share/conll2009/jsonl/CoNLL2009_sample.10k.conll2009 \
)

DATASET_PATHS=(\
    data/conll2009/train.conll2009 \
    data/conll2009/conll2009_10-percent/train.conll2009 \
    data/conll2009/conll2009_25-percent/train.conll2009 \
    data/conll2009/conll2009_50-percent/train.conll2009 \
    data/conll2009/conll2009_75-percent/train.conll2009 \
    data/conll2009/validation.conll2009 \
    data/conll2009/test.conll2009 \
    data/conll2009/test-ood.conll2009 \
    data/conll2009/sample.200.conll2009 \
    data/conll2009/sample.10k.conll2009 \
)

len=${#INPUT_PATHS[@]}

for (( i=0; i<$len; i++ )); do
    INPUT=${INPUT_PATHS[$i]}
    OUTPUT=${OUTPUT_PATHS[$i]}
    DATASET=${DATASET_PATHS[$i]}
    python scripts/preprocessing/preprocess_conll2009_data.py \
        --input $INPUT \
        --output $OUTPUT \
        --frames $FRAMES_PATH \
        --argm_definitions $ARGM_DEFINITIONS_PATH
    cp $OUTPUT $DATASET
done

OUTPUT_DIR=data/compositional/conll2009
mkdir -p $OUTPUT_DIR

# produce train
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/train.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/train.srl

# produce validation
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/validation.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/validation.srl

# produce test
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/test.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/test.srl

# produce test-ood
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/test-ood.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/test-ood.srl

# produce train (10%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/conll2009_10-percent/train.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/10-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/10-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/10-percent/test.srl
cp $OUTPUT_DIR/test-ood.srl $OUTPUT_DIR/10-percent/test-ood.srl

# produce train (25%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/conll2009_25-percent/train.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/25-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/25-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/25-percent/test.srl
cp $OUTPUT_DIR/test-ood.srl $OUTPUT_DIR/25-percent/test-ood.srl

# produce train (50%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/conll2009_50-percent/train.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/50-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/50-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/50-percent/test.srl
cp $OUTPUT_DIR/test-ood.srl $OUTPUT_DIR/50-percent/test-ood.srl

# produce train (75%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/conll2009_75-percent/train.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/75-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/75-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/75-percent/test.srl
cp $OUTPUT_DIR/test-ood.srl $OUTPUT_DIR/75-percent/test-ood.srl

printf "Done!\n\n"
