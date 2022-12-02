#!/bin/bash

FRAMES_PATH=data_share/conll2012/frames/conll2012_frames.json
ARGM_DEFINITIONS_PATH=data_share/conll2012/frames/conll2012_argm_definitions.json

INPUT_PATHS=(\
    data_share/conll2012/original/CoNLL2012_train.txt \
    data_share/conll2012/original/CoNLL2012_train_10-percent.txt \
    data_share/conll2012/original/CoNLL2012_train_25-percent.txt \
    data_share/conll2012/original/CoNLL2012_train_50-percent.txt \
    data_share/conll2012/original/CoNLL2012_train_75-percent.txt \
    data_share/conll2012/original/CoNLL2012_dev.txt \
    data_share/conll2012/original/CoNLL2012_test.txt \
    data_share/conll2012/original/CoNLL2012_sample.txt \
)

OUTPUT_PATHS=(\
    data_share/conll2012/jsonl/CoNLL2012_train.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_train_10-percent.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_train_25-percent.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_train_50-percent.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_train_75-percent.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_dev.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_test.conll2012 \
    data_share/conll2012/jsonl/CoNLL2012_sample.conll2012 \
)

DATASET_PATHS=(\
    data/conll2012/train.conll2012 \
    data/conll2012/conll2012_10-percent/train.conll2012 \
    data/conll2012/conll2012_25-percent/train.conll2012 \
    data/conll2012/conll2012_50-percent/train.conll2012 \
    data/conll2012/conll2012_75-percent/train.conll2012 \
    data/conll2012/validation.conll2012 \
    data/conll2012/test.conll2012 \
    data/conll2012/sample.conll2012 \
)

len=${#INPUT_PATHS[@]}

for (( i=0; i<$len; i++ )); do
    INPUT=${INPUT_PATHS[$i]}
    OUTPUT=${OUTPUT_PATHS[$i]}
    DATASET=${DATASET_PATHS[$i]}
    python scripts/preprocessing/preprocess_conll2012_data.py \
        --input $INPUT \
        --output $OUTPUT \
        --frames $FRAMES_PATH \
        --argm_definitions $ARGM_DEFINITIONS_PATH
    cp $OUTPUT $DATASET
done

OUTPUT_DIR=data/compositional/conll2012
mkdir -p $OUTPUT_DIR

# produce train
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/train.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/train.srl

# produce validation
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/validation.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/validation.srl

# produce test
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/test.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/test.srl

# produce train (10%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/conll2012_10-percent/train.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/10-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/10-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/10-percent/test.srl

# produce train (25%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/conll2012_25-percent/train.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/25-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/25-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/25-percent/test.srl

# produce train (50%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/conll2012_50-percent/train.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,span-srl \
    --output-path $OUTPUT_DIR/50-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/50-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/50-percent/test.srl

# produce train (75%)
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2012/conll2012_75-percent/train.conll2012 \
    --datasets-identifier conll2012 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/75-percent/train.srl
cp $OUTPUT_DIR/validation.srl $OUTPUT_DIR/75-percent/validation.srl
cp $OUTPUT_DIR/test.srl $OUTPUT_DIR/75-percent/test.srl

printf "Done!\n\n"
