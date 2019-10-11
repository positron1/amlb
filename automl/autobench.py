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
debugmode = False
<<<<<<< HEAD
#True 
=======
>>>>>>> dc50722742c88a5c9cce05335e618e109892f54f
if debugmode:
  pass
else:
  orig_stdout = sys.stdout
current_time = DateTime(time.time(), 'US/Eastern')

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
task ='bre'
if task=='bt':
   datalist = glob.glob(dirt+"opentest/*sas7bdat")
   metalist = glob.glob(dirt+"meta/*csv")
   datalist = remove_dirt(datalist,dirt+'/opentest/')
   metalist = remove_dirt(metalist,dirt+'/meta/')
elif task=='bre':
   datalist = glob.glob(dirt+"binaryRareEvent/data/*sas7bdat")
   metalist = glob.glob(dirt+"binaryRareEvent/meta/*csv")
   datalist = remove_dirt(datalist,dirt+'/binaryRareEvent/data/')
   metalist = remove_dirt(metalist,dirt+'/binaryRareEvent/meta/')
   dirt = dirt+'binaryRareEvent/'
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
runlist =['5','6','7','8','9']
rep= 5
timelist = [900]
foldlist = [0]

prep = False
runlist = ['3']
runlist =['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
runlist = ['4']

runlist =['0','1','2','3','4']
timestamp = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
        str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
logfile = open('results/log_'+str(len(runlist))+'dataset'+str(timelist[0])+str(foldlist[0])+"rep"+str(rep)+str(timestamp)+".txt",'w')

if debugmode:
  pass
else:
  sys.stdout = logfile

for im,meta in enumerate(metalist):
    myid = meta.split('_')[0]
    if myid[2:] in runlist:
      print(myid[2:])
      framework = 'autosklearn'
      ncore = 15
      dataset = datalist[im]# "uci_bank_marketing_pd"
      print("\ndataset:\t",dataset)
      print("\nmetadata information:\t",meta)
      try:
        runbenchmark(prep,dataset,framework,foldlist,ncore,timelist,dirt,meta,fitmetrics,rep,logfile)
      except:
        print('Failed:\t',myid,dataset)
        continue
if debugmode:
  pass
else:   
  sys.stdout = orig_stdout
  logfile.close()
