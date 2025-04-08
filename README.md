# Enhancing High-density Spatial Transcriptomics from Histology Images using HisHRST
Spatial transcriptomics (ST) is a technology that integrates spatial information with gene expression analysis to study
the spatial distribution of genes within tissues and their regulatory mechanisms. However, it is limited by the sparsity
of sequencing spots and the high cost of spatial transcriptomics technology, which hinders its widespread application in
biomedical research. An alternative and more cost-effective strategy is to leverage deep learning methods to infer high-
density gene expression profiles from histological images. To this end, we developed the HisHRST method based on an
offline pathological image foundation model, aiming to accurately generate high-density spatial transcriptomic data from
histological images. This method employs a multi-head attention mechanism to incorporate spatial location information,
thereby enhancing feature representation. We systematically evaluated HisHRST on six ST datasets and compared its
performance with five existing methods. Experimental results demonstrate that HisHRST can accurately predict gene
expression profiles for unmeasured spots, refine gene expression patterns, and effectively preserve the original spatial
structure of gene expression. Furthermore, this method facilitates the identification of biologically meaningful pathways,
thereby advancing the understanding of key biological processe.


## Overview of HRST

![](model.png)

## Setup
```
pip install -r requirement.txt
```

## Data
All the datasets used in this paper can be downloaded from url：https://zenodo.org/records/12792074


## Running Experiments

To train the neural network, use the following command:

```
python train.py
```

To evaluate the model, run:

```
python test.py
```

## Contact details

If you have any questions, please contact zhicengshi@stu.ynu.edu.com and wenwen.min@qq.com
