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
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss,f1_score
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

from DateTime import DateTime
import time

numeric_features =[]
categorical_features =[]
dirt = '../data/'
datalist = glob.glob(dirt+"opentest/*sas7bdat")
metalist = glob.glob(dirt+"meta/*csv")
datalist = remove_dirt(datalist,dirt+'/opentest/')
metalist = remove_dirt(metalist,dirt+'/meta/')
print(datalist)
print(metalist)
fitmetrics = autosklearn.metrics.log_loss
datalist =sorted(datalist)
metalist = sorted(metalist)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#for _ in range(5):
for im,meta in enumerate(metalist):
    current_time = DateTime(time.time(), 'US/Eastern')
    if meta[:4]=='id15':
      framework = 'autosklearn'
      current_time = DateTime(time.time(), 'US/Eastern')
      prepart = True
      ncore = 4
      dataset = datalist[im]# "uci_bank_marketing_pd"
      print("\ndataset:\t",dataset)
      print("\nmetadata information:\t",meta)
      for foldn in [10]:
        for timeforjob in [100]:
          runbenchmark(dataset,framework,foldn,ncore,timeforjob,dirt,meta,fitmetrics)
      
