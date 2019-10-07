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

orig_stdout = sys.stdout
current_time = DateTime(time.time(), 'US/Eastern')

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
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
runlist =['0','1','2','3','4','10','11','12','13','14']
runlist =['12','13','14']
rep= 4 
timelist = [900]
foldlist = [10]
timestamp = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
        str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
logfile = open('results/log_'+str(len(runlist))+'dataset'+str(timelist[0])+str(foldlist[0])+"rep"+str(rep)+str(timestamp)+".txt",'w')
sys.stdout = logfile
for im,meta in enumerate(metalist):
    myid = meta.split('_')[0]
    if myid[2:] in runlist:
      print(myid[2:])
      framework = 'autosklearn'
      prepart = True
      ncore = 4
      dataset = datalist[im]# "uci_bank_marketing_pd"
      print("\ndataset:\t",dataset)
      print("\nmetadata information:\t",meta)
      runbenchmark(dataset,framework,foldlist,ncore,timelist,dirt,meta,fitmetrics,rep,logfile)
     
sys.stdout = orig_stdout
logfile.close()
