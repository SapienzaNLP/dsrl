<div align="center">    
 
# Semantic Role Labeling Meets Definition Modeling: Using Natural Language to Describe Predicate-Argument Structures     

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)]()
[![Conference](http://img.shields.io/badge/EMNLP-2022-4b44ce.svg)](https://2022.emnlp.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## About the project
This is the repository for the paper [*Semantic Role Labeling Meets Definition Modeling: Using Natural Language to Describe Predicate-Argument Structures*](), published in Findings of EMNLP 2022 and authored by [Simone Conia](https://c-simone.github.io/), [Edoardo Barba](https://edobobo.github.io/), Alessandro Scirè, and [Roberto Navigli](https://www.diag.uniroma1.it/navigli/).

## Abstract
> One of the common traits of past and present approaches for Semantic Role Labeling (SRL) is that they rely upon discrete labels drawn from a predefined linguistic inventory to classify predicate senses and their arguments. However, we argue this need not be the case. In this paper, we present an approach that leverages Definition Modeling to introduce a generalized formulation of SRL as the task of describing predicate-argument structures using natural language definitions instead of discrete labels. Our novel formulation takes a first step towards placing interpretability and flexibility foremost, and yet our experiments and analyses on PropBank-style and FrameNet-style, dependency-based and span-based SRL also demonstrate that a flexible model with an interpretable output does not necessarily come at the expense of performance.

## Cite this work
If you use any part of this work, please consider citing the paper as follows:

```
@inproceedings{conia-etal-2022-dsrl,
    title     = "{S}emantic {R}ole {L}abeling Meets Definition Modeling: {U}sing Natural Language to Describe Predicate-Argument Structures",
    author    = "Conia, Simone and Barba, Edoardo and Scir\`e, Alessandro and Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month     = dec,
    year      = "2022",
    address   = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
}
```

## How to use
You'll need a working Python environment to run the code. The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/download/) which provides the `conda` package manager. Anaconda can be installed in your user directory and does not interfere with the system Python installation. 

We use `conda` virtual environments to manage the project dependencies in
isolation. Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command and follow the steps to create a separate environment:
```bash
> bash setup.sh
> Enter environment name (recommended: multilingual-srl): dsrl
> Enter python version (recommended: 3.8): 3.8
> Enter torch version (recommended 1.9.0): 1.9
> Enter cuda version (e.g. '11.1' or 'none' to avoid installing cuda support):
```
All the code in this repository was tested using Python 3.8 and CUDA 11.1.

### Getting the data
Depending on the task you want to perform (e.g., dependency-based SRL or span-based SRL),
you need to obtain some datasets (unfortunately, some of these datasets require a license fee).
> NOTE: Not all of the following datasets are required. E.g., if you are only interested
  in dependency-based SRL with PropBank labels, you just need CoNLL-2009. 

* [Hajic et al., 2009. The CoNLL-2009 Shared Task: Syntactic and Semantic Dependencies in Multiple Languages](https://aclanthology.org/W09-1201/).
The dataset is available on LDC ([LDC2012T04](https://catalog.ldc.upenn.edu/LDC2012T04)).
* [Pradhan et al., 2012. CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes](https://aclanthology.org/W12-4501/).
The dataset is available on LDC ([LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)).
* [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/).

Once you have downloaded and unzipped the data, place it in `data_share/name-of-the-dataset/original`.

### Data preprocessing
To preprocess the datasets, run the script `preprocess_<dataset_name>.sh` from the root directory of the project. For example, for CoNLL-2009:
```bash
bash scripts/preprocessing/preprocess_conll2009_data.sh
```

### Training a model
Once you have everything ready, training a model is quite simple. Just run the command:
```bash
EXPERIMENT_NAME=large_conll2009

classy train generation data/compositional/conll2009 \
    -n $EXPERIMENT_NAME \
    --profile large_conll2009 \
    --fp16 \
    --wandb dsrl-emnlp@$EXPERIMENT_NAME \
    -c \
        callbacks=evaluation \
        callbacks.0.settings.0.prediction_param_conf_path=configurations/prediction-params/beam.yaml \
        callbacks.0.settings.0.limit=100000 \
        callbacks.0.settings.0.token_batch_size=4096
```

You can take a look in `scripts/training` for a few examples of how to train the model with different configurations.

## Acknowledgements

The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE No. 726487](http://mousse-project.org/) under the European Union’s Horizon 2020 research and innovation programme.


## License
This work (the paper and all the contents of this repository) are licensed under Creative Commons Attribution-NonCommercial 4.0 International.

See LICENSE.txt for more details.
