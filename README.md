# Set2Gaussian
> Set2Gaussian: Embedding Gene Sets as Gaussian Distributions for Large-scale Gene Set Analysis

## Introduction
Set2Gaussian is a network-based embedding approach which takes a set of gene sets as input and output an embedding representation for each gene set. It could embed more than 10,000 gene sets. Instead of embedding as single points, Set2Gaussian embeds each gene set as a Gaussian distribution in order to model the diverse functions within a gene set.

## Publication (under review)

**Set2Gaussian: Embedding Gene Sets as Gaussian Distributions for Large-scale Gene Set Analysis**.
Sheng Wang, Emily Flynn, Russ B. Altman.

## Dataset
We provide the dataset and embeddings of 13,886 gene sets from NCI, Reactome, and MSigDB [figshare](https://figshare.com/projects/Set2Gaussian/71099)

## How to run

An example is in src/Grep_run_all_methods.py
```
cd src
python Grep_run_all_methods.py
```

## Prerequisite
* python 3.6 (with slight modification, python 2.7 can also be used to run our tool)
* python packages (numpy 1.14+, scipy 1.1+, networkx 2.3+, tensorflow 1.14.0)

## Questions
For questions about the data and code, please contact swang91@stanford.edu. We will do our best to provide support and address any issues. We appreciate your feedback!
