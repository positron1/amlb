### This is for Yonglin's person use. All rights reserved.
### Contact: zhuygln@gmail.com
### Log of progress and to-dos.


# Week of 061719



## Summary
- Setup environment for several open source auto ML tools on window sublinux and linux server(redhat). For now, our list of tools are auto-sklearn, tpot, h2o.ai. I some did some research on how the automated ML benchmarking was done in this field. We may be able to introduce more tools without much extra work.

- For now, our list of tools are auto-sklearn, tpot, h2o.ai. I some did some research on how the automated ML benchmarking was done in this field. We may be able to introduce more tools without much extra work.

- Suggested examples from the tutorials are under testing. I chose classification problems with more focus on banking and rare event classification. 



# Week of 062419
# Environment 
## Server:
- Virtual environment with conda: autobench(autobenchmark tool), autosklearn, tpot, autoweka, h2o,


## Sublinux:

- Vritual environment with conda
- yz_automl, yz-autosklearn(autosklearn), yz_h2o(h2o), yz_tpot(TPOT), yzwork(autobenchmark),


## Window:

- vritual environment with conda. (check: conda env list)
- automl, 

# Test 0 
## Dateset: bank marketing (https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Abstract: The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
- Data Set Characteristics:  Multivariate
- Number of Instances: 45211
- Area: Business
- Attribute Characteristics: Real
- Number of Attributes: 17
- Associated Tasks: Classification
- Missing Values? N/A

## Metric
- AUC

## open source frameworks

### autosklearn
Efficient and Robust Automated Machine Learning, Feurer et al., Advances in Neural Information Processing Systems 28 (NIPS 2015).
- parameter

### H2o
### oboe

### autoweka

## Model Studio
- automl, h

## Week of 070818
- use show_model to interpret the model and parameter selection of autosklearn
## Week of 071519
- use pickle(https://dataaspirant.com/2017/02/13/save-scikit-learn-models-with-python-pickle/) to save modeli
- use netron to visualize final model???(https://github.com/lutzroeder/netron/tree/v3.2.8)
- use train data with 3 fold cross validation and test hold-out
todo: 
try to do 3 fold and visualize

score from autosklearn and tpot: avg(CV)
=======
- add read data method in utils
- one hot encoding for train and test: 
- use pandas for onehotencoding
## Week of 072219
- Use vanilla_bank, let framework do the partition 
- Use prepartitioned data for benchmarking

####################
- automated detect numerical/categ features
- why getting 0.73 for bank marking data: 1. is this data specific? try adult. 2. 
- automated the experiment for time/fold/core.
- automlbenchmark do the cv resample but we want to let the framework do it by itself.
- based on the automl benchmark? what is been done for cross validation? 
- try fit_ensemble for autosklearn
- try default metric in fit() of autosklearn
- how to manage open source packages in MLPA
- feat_type  
-
######################################################## 
08/12/19
- lightgbm with banking, ~ validate 0.94
- exfold_bank -- metric auc
- adult with auto detect 1800
 
- 
