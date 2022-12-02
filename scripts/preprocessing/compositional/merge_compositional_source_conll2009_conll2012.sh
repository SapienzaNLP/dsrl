mkdir data/conll2009_conll2012_compositional_source

# merge train
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/train.conll2009 data/conll2012/train.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,span-srl propbank,dep-srl \
  --apply-to-source \
  --output-path data/conll2009_conll2012_compositional_source/train.srl

# merge validation
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/validation.conll2009 data/conll2012/validation.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,span-srl propbank,dep-srl \
  --apply-to-source \
  --output-path data/conll2009_conll2012_compositional_source/validation.srl

# merge test
PYTHONPATH=. python scripts/preprocessing/merge_datasets.py \
  --input-datasets data/conll2009/test.conll2009 data/conll2012/test.conll2012 \
  --datasets-identifier conll2009 conll2012 \
  --compositional-tokens propbank,span-srl propbank,dep-srl \
  --apply-to-source \
  --output-path data/conll2009_conll2012_compositional_source/test.srl
