OUTPUT_DIR=data/conll2009_compositional

mkdir $OUTPUT_DIR

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
