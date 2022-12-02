#!/bin/bash

## ALL
cat data/compositional/conll2009/10-percent/train.srl \
    data/compositional/conll2012/10-percent/train.srl \
    data/compositional/framenet17/10-percent/train.srl \
    > data/compositional/all/10-percent/train.srl

cat data/compositional/conll2009/10-percent/validation.srl \
    data/compositional/conll2012/10-percent/validation.srl \
    data/compositional/framenet17/10-percent/validation.srl \
    > data/compositional/all/10-percent/validation.srl

cat data/compositional/conll2009/10-percent/test.srl \
    data/compositional/conll2012/10-percent/test.srl \
    data/compositional/framenet17/10-percent/test.srl \
    > data/compositional/all/10-percent/test.srl

## CONLL2009 + CONLL2012
cat data/compositional/conll2009/10-percent/train.srl \
    data/compositional/conll2012/10-percent/train.srl \
    > data/compositional/conll2009+conll2012/10-percent/train.srl

cat data/compositional/conll2009/10-percent/validation.srl \
    data/compositional/conll2012/10-percent/validation.srl \
    > data/compositional/conll2009+conll2012/10-percent/validation.srl

cat data/compositional/conll2009/10-percent/test.srl \
    data/compositional/conll2012/10-percent/test.srl \
    > data/compositional/conll2009+conll2012/10-percent/test.srl

## CONLL2009 + Framenet17
cat data/compositional/conll2009/10-percent/train.srl \
    data/compositional/framenet17/10-percent/train.srl \
    > data/compositional/conll2009+framenet17/10-percent/train.srl

cat data/compositional/conll2009/10-percent/validation.srl \
    data/compositional/framenet17/10-percent/validation.srl \
    > data/compositional/conll2009+framenet17/10-percent/validation.srl

cat data/compositional/conll2009/10-percent/test.srl \
    data/compositional/framenet17/10-percent/test.srl \
    > data/compositional/conll2009+framenet17/10-percent/test.srl

## CONLL2012 + Framenet17
cat data/compositional/conll2012/10-percent/train.srl \
    data/compositional/framenet17/10-percent/train.srl \
    > data/compositional/conll2012+framenet17/10-percent/train.srl

cat data/compositional/conll2012/10-percent/validation.srl \
    data/compositional/framenet17/10-percent/validation.srl \
    > data/compositional/conll2012+framenet17/10-percent/validation.srl

cat data/compositional/conll2012/10-percent/test.srl \
    data/compositional/framenet17/10-percent/test.srl \
    > data/compositional/conll2012+framenet17/10-percent/test.srl
