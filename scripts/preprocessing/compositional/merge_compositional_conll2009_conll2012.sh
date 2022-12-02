mkdir data/conll2009_conll2012_compositional

# merge train
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/train.conll2009 data/conll2012/train.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,dep-srl propbank,span-srl \
  --output-path data/conll2009_conll2012_compositional/train.srl

# merge validation
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/validation.conll2009 data/conll2012/validation.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,dep-srl propbank,span-srl \
  --output-path data/conll2009_conll2012_compositional/validation.srl

# merge test
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/test.conll2009 data/conll2012/test.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,dep-srl propbank,span-srl \
  --output-path data/conll2009_conll2012_compositional/test.srl

# merge test
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/test-ood.conll2009 \
  --datasets-identifier conll2009 \
  --compositional-tokens propbank,dep-srl \
  --output-path data/conll2009_conll2012_compositional/test-ood.srl
