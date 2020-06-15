# Set2Gaussian
> Set2Gaussian: Embedding Gene Sets as Gaussian Distributions for Large-scale Gene Set Analysis

## Introduction
Set2Gaussian is a network-based embedding approach which takes a set of gene sets as input and output an embedding representation for each gene set. It could embed more than 10,000 gene sets. Instead of embedding as single points, Set2Gaussian embeds each gene set as a Gaussian distribution in order to model the diverse functions within a gene set.

## Publication (under review)

**Set2Gaussian: Embedding Gene Sets as Gaussian Distributions for Large-scale Gene Set Analysis**.
Sheng Wang, Emily Flynn, Russ B. Altman.

## Dataset
We provide the dataset and embeddings of 13,886 gene sets from NCI, Reactome, and MSigDB [figshare](https://figshare.com/projects/Set2Gaussian/71099)
A sample dataset is in the data folder.
network.txt is the network in the following format:

node1	node2	weight

...

node_set.txt is the node set in the following format:

set1	node1

set1	node2

...

## How to run

An example is in src/Grep_run_all_methods.py. It takes data/network.txt and data/node_set.txt as input. First replace them with your network and gene sets.
```
cd src
python Grep_run_all_methods.py
```
The embeddings will be saved in output_embed

## Prerequisite
* python 2.7 (with slight modification, python 3.6 can also be used to run our tool)
* python packages (numpy 1.14+, scipy 1.1+, networkx 2.3+, tensorflow 1.14.0)

## Questions
For questions about the data and code, please contact swang91@stanford.edu. We will do our best to provide support and address any issues. We appreciate your feedback!
