#!/bin/bash

FRAMES_PATH=data_share/framenet15/frames/framenet_frames.json

INPUT_PATHS=(\
    data_share/framenet15/original/fn1.5.fulltext.train.syntaxnet.conll \
    data_share/framenet15/original/fn1.5.exemplar.train.syntaxnet.conll \
    data_share/framenet15/original/fn1.5.dev.syntaxnet.conll \
    data_share/framenet15/original/fn1.5.test.syntaxnet.conll \
)

OUTPUT_PATHS=(\
    data_share/framenet15/jsonl/FrameNet15_train.fulltext.framenet15 \
    data_share/framenet15/jsonl/FrameNet15_train.exemplar.framenet15 \
    data_share/framenet15/jsonl/FrameNet15_dev.framenet15 \
    data_share/framenet15/jsonl/FrameNet15_test.framenet15 \
)

DATASET_PATHS=(\
    data/framenet15/train.framenet15 \
    data/framenet15/validation.framenet15 \
    data/framenet15/test.framenet15 \
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

cp data_share/framenet15/jsonl/FrameNet15_dev.framenet15 data/framenet15/validation.framenet15
cp data_share/framenet15/jsonl/FrameNet15_test.framenet15 data/framenet15/test.framenet15
cat data_share/framenet15/jsonl/FrameNet15_train.fulltext.framenet15 data_share/framenet15/jsonl/FrameNet15_train.exemplar.framenet15 > data/framenet15/train.framenet15

OUTPUT_DIR=data/compositional/framenet15
mkdir -p $OUTPUT_DIR

# produce train
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet15/train.framenet15 \
    --datasets-identifier framenet15 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/train.srl

# produce validation
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet15/validation.framenet15 \
    --datasets-identifier framenet15 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/validation.srl

# produce test
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/framenet15/test.framenet15 \
    --datasets-identifier framenet15 \
    --compositional-tokens framenet,span-srl \
    --output-path $OUTPUT_DIR/test.srl

printf "Done!\n\n"
