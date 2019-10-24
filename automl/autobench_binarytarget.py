import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
###### Read in data
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, vstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.model_selection import cross_val_score
##################################################
import json
import jsonpickle
from sas7bdat import SAS7BDAT
import pandas as pd
import os
import sys
import logging
import optparse
from utils import *
from runbench import *
import secrets

from postprocessing import *
from DateTime import DateTime
import time
from collections import Counter


logmode = False
if logmode:
    orig_stdout = sys.stdout

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

##################################################################################
###             Inputs                                  #
#################################################################################
framework = 'autosklearn'
task = 'bt'  # interval target task
prep = False  # Data preprocessing with meta data
dirt = '/root/data/'  # dataset directory
outputdir = './results/'
task_token = secrets.token_hex(8)  # generate unique token for this run
#################################################################################
runlist = ['5', '7']  # dataset id #
rep = 2  # repetition
metalearning = True  # default for autosklearn
corelist = [16]
foldlist = [0]  # 0: single validation, no cross validation
timelist = [100]  # time limit for training in seconds
#################################################################################
# Initial setup
#################################################################################
dirt, logfile, csvdatalist, sasdatalist, metalist, timestamp, fitmetrics = init(
    dirt, task, runlist, timelist, foldlist, rep, task_token)
metadataid = get_id(metalist)
csvdataid = get_id(csvdatalist)
sasdataid = get_id(sasdatalist)
print('csvdataid\n', csvdataid)
print('sasdataid\n', sasdataid)
print('metadataid\n', metadataid)

if logmode:
    sys.stdout = logfile
#################################################################################
# runing ...
#################################################################################


for ind in runlist:
    dataset,meta=check_id(ind,csvdataid,sasdataid,metadataid)
    try:
        runbenchmark(metalearning, prep, dataset, framework, foldlist, corelist,
                     timelist, dirt, meta, fitmetrics, rep, logfile, outputdir,task_token)
    except:
        print('Failed:\t', ind)  # ,dataset)
        traceback.print_exc(file=sys.stdout)
        continue
#################################################################################
# Summary of results
#################################################################################
for lf, locfold in enumerate(timelist):
    ldirt = 'results/'+str(locfold)+'s/'
    dataname, auclist, loglosslist, acclist = get_results_clf(
        ldirt, timestamp, task_token)

if logmode:
    sys.stdout = orig_stdout
    logfile.close()
