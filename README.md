# ML Reproducibility Challenge 2021 

Reproducement of the paper:
> [Learning Unknown from Correlations: Graph Neural Network for Inter-novel-protein Interaction Prediction](https://arxiv.org/abs/2105.06709) \
> Authors: Guofeng Lv, Zhiqiang Hu, Yanguang Bi, Shaoting Zhang \
> Arxiv extended verison (arxiv: https://arxiv.org/abs/2105.06709)


## Abstract of the original paper

The study of multi-type Protein-Protein Interaction (PPI) is fundamental for understanding biological processes from a systematic perspective and revealing disease mechanisms. Existing methods suffer from significant performance degradation when tested in unseen dataset. In this paper, we investigate the problem and find that it is mainly attributed to the poor performance for inter-novel-protein interaction prediction. However, current evaluations overlook the inter-novel-protein interactions, and thus fail to give an instructive assessment. As a result, we propose to address the problem from both the evaluation and the methodology. Firstly, we design a new evaluation framework that fully respects the inter-novel-protein interactions and gives consistent assessment across datasets. Secondly, we argue that correlations between proteins must provide useful information for analysis of novel proteins, and based on this, we propose a graph neural network based method (GNN-PPI) for better inter-novel-protein interaction prediction. Experimental results on real-world datasets of different scales demonstrate that GNN-PPI significantly outperforms state-of-the-art PPI prediction methods, especially for the inter-novel-protein interaction prediction.

## Reproducibility summary

### Scope of Reproducibility

We first inspect if the evaluation proposed is objectively better, and not just useful for the authors to show the superiority of their model, 
and secondly, we try to reproduce the results of the proposed model in comparison with previous state-of-the-art, [PIPR](https://github.com/muhaochen/seq_ppi).

### Methodology

For the reproduction we used authors \href{https://github.com/lvguofeng/GNN_PPI}{code}, slightly changing the pipeline for automatization. We also used \href{https://github.com/muhaochen/seq_ppi}{PIPR code}, where we completely changed the pipeline, to be able to use it on the same datasets as GNN-PPI, but used their function for building the model. 
The experiments were run on Nvidia Titan X GPU, using around 250 GPU hours altogether.

### Results
We reproduced the papers results within standard deviations of our repeated experiments. 
In some cases, this still means there is a big difference between the performances, which is coming from different 
train-test splits of the newly proposed splitting schemes. Even with these discrepancies we still managed to confirm all authors claims. 
The proposed model GNN-PPI performed better than PIPR overall and for inter-novel-protein interactions, evaluation on 
the proposed schemes predicted the generalization performance better, and their model is also robust for predictions 
for newly discovered proteins -- here our results were surprising, they were even better when the network was built knowing fewer proteins.
More detailed results are described in the report.

### What was easy

It was easy to run GNN-PPI code on different datasets and with different parameters, as their repository is nicely organized and the code is clearly structured. 
It was also easy to understand their idea of the problem, the reasons for new evaluation and the framework of their proposed model.

### What was difficult

In both models used in this reproduction, the environment setup was harder than expected. There was no documentation or 
comments in the code, which made it hard at first to understand it. Some debugging was needed for GNN-PPI and a lot of code changes for PIPR to train well.
To help with this in further reproducement, we uploaded the environment files, so you can directly build the correct 
environment and we also packed PIPR model into functions, so you can just set the parameters as you wish and then call the functions.

___________________________________________

## Further reproduction
To further reproduce the paper, or just to use the models described here, you can folow this instructions.

This repository contains:
- Environment Setup
- Data Processing
- Training and testing for GNN-PPI
- Training and testing for PIPR
- Combining test results
- Reproducibility report

### Environment Setup

Environment files are available in folder *environment*. Use environment from ``gnn_ppi.yml`` to run the *GNN-PPI* model from the paper 
and the one from ``pipr.yml`` to run the baseline model, PIPR.

#### Dataset Download:

All datasets are available on this repository, except for the big dataset of all interactions (`9606.protein.actions.all_connected.txt`).
You can download it from [Google Drive](https://drive.google.com/drive/folders/1yXYhnUrkzQBmdq1mVq7P6sgrQnX_rZKU?usp=sharing).

Or if it is easier for you, here is the path to authors Baidu download:
- https://pan.baidu.com/s/1FU-Ij3LxyP9dOHZxO3Aclw (Extraction code: tibn)

After downloading, save it to folder `data`.

### Data Processing

The data processing codes are in gnn_data.py (Class GNN_DATA), as were prepared by authors.
This is all automatically done when training the model. To prepare data for PIPR, you need to first train the GNN-PPI 
models, as their data splitting is used for creating same datasets for PIPR. After running GNN-PPI training, you should 
run `prepare_data_PIPR`, to build the datasets that will be further used for PIPR training.

### Training and testing

#### GNN-PPI
Training codes in `gnn_train.py`, and the run script in `run.py`. You can set all parameters for running on different 
datasets, split methods and with different seeds in the `run.py` script.
Similarly for testing, you can set everything in `run_test.py` (used for normal testing) and `run_test_bigger.py` 
(used for testing generalization) scripts.

#### PIPR
Training and testing of PIPR are both implemented in `PIPR/type/model/lasanga/rcnn.py`. You can set parameters there, 
and run it to run both training and testing.

### Combining results
Script `combine_results.py` will read the results of different models, calculate their mean and standard deviation and 
write them in a latex table, that we used as a skeleton for tables in our report.
