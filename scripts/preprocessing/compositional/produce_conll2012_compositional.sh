OUTPUT_DIR=data/conll2012_compositional

mkdir $OUTPUT_DIR

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
