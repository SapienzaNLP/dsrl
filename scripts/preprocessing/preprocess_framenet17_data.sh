#!/bin/bash

FRAMES_PATH=data_share/framenet17/frames/framenet_frames.json

INPUT_PATHS=(\
    data_share/framenet17/original/fn1.7.fulltext.train.syntaxnet.conll \
    data_share/framenet17/original/fn1.7.exemplar.train.syntaxnet.conll \
    data_share/framenet17/original/fn1.7.dev.syntaxnet.conll \
    data_share/framenet17/original/fn1.7.test.syntaxnet.conll \
)

OUTPUT_PATHS=(\
    data_share/framenet17/jsonl/FrameNet17_train.fulltext.framenet17 \
    data_share/framenet17/jsonl/FrameNet17_train.exemplar.framenet17 \
    data_share/framenet17/jsonl/FrameNet17_dev.framenet17 \
    data_share/framenet17/jsonl/FrameNet17_test.framenet17 \
)

DATASET_PATHS=(\
    data/framenet17/train.framenet17 \
    data/framenet17/validation.framenet17 \
    data/framenet17/test.framenet17 \
)

len=${#INPUT_PATHS[@]}

for (( i=0; i<$len; i++ )); do
    INPUT=${INPUT_PATHS[$i]}
    OUTPUT=${OUTPUT_PATHS[$i]}
    DATASET=${DATASET_PATHS[$i]}
    python scripts/preprocessing/preprocess_framenet_data.py \
        --input $INPUT \
        --frames $FRAMES_PATH \
        --output $OUTPUT
done

cp data_share/framenet17/jsonl/FrameNet17_dev.framenet17 data/framenet17/validation.framenet17
cp data_share/framenet17/jsonl/FrameNet17_test.framenet17 data/framenet17/test.framenet17
cat data_share/framenet17/jsonl/FrameNet17_train.fulltext.framenet17 data_share/framenet17/jsonl/FrameNet17_train.exemplar.framenet17 > data/framenet17/train.framenet17

OUTPUT_DIR=data/compositional/framenet17
mkdir -p $OUTPUT_DIR

# produce train
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet17/train.framenet17 \
    --datasets-identifier framenet17 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/train.srl

# produce validation
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet17/validation.framenet17 \
    --datasets-identifier framenet17 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/validation.srl

# produce test
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet17/test.framenet17 \
    --datasets-identifier framenet17 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/test.srl

printf "Done!\n\n"
