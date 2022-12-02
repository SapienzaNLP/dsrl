OUTPUT_DIR=data/compositional/all

mkdir -p $OUTPUT_DIR

# merge train
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/train.conll2009 data/conll2012/train.conll2012 data/framenet17/train.framenet17 \
    --datasets-identifier conll2009 conll2012 framenet17 \
    --compositional-tokens propbank,dep-srl propbank,span-srl framenet,span-srl \
    --output-path $OUTPUT_DIR/train.srl

# merge validation
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/validation.conll2009 data/conll2012/validation.conll2012 data/framenet17/validation.framenet17 \
    --datasets-identifier conll2009 conll2012 framenet17 \
    --compositional-tokens propbank,dep-srl propbank,span-srl framenet,span-srl \
    --output-path $OUTPUT_DIR/validation.srl

# merge test
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/test.conll2009 data/conll2012/test.conll2012 data/framenet17/test.framenet17 \
    --datasets-identifier conll2009 conll2012 framenet17 \
    --compositional-tokens propbank,dep-srl propbank,span-srl framenet,span-srl \
    --output-path $OUTPUT_DIR/test.srl

# merge test-ood
PYTHONPATH=. python scripts/preprocessing/compositional/merge_datasets.py \
    --input-datasets data/conll2009/test-ood.conll2009 \
    --datasets-identifier conll2009 \
    --compositional-tokens propbank,dep-srl \
    --output-path $OUTPUT_DIR/test-ood.srl
