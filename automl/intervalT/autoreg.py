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
from postprocessing import *
from runbench import *
import secrets 
from DateTime import DateTime

import time
debugmode = True 
if debugmode:
  pass
else:
  orig_stdout = sys.stdout
def init(dirt,task,runlist,timelist,foldlist,rep,task_token):
    current_time = DateTime(time.time(), 'US/Eastern')
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
    elif task =='it':
       datalist = glob.glob(dirt+"intervalTarget/data/*sas7bdat")
       metalist = glob.glob(dirt+"intervalTarget/meta/*csv")
       datalist = remove_dirt(datalist,dirt+'/intervalTarget/data/')
       metalist = remove_dirt(metalist,dirt+'/intervalTarget/meta/')
       dirt = dirt+'intervalTarget/'
       fitmetrics = autosklearn.metrics.mean_squared_error
    
    datalist =sorted(datalist)
    metalist = sorted(metalist)
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    timestamp = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
            str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
    logfile = open('results/log_'+str(len(runlist))+'dataset_'+str(timelist[0])+'s_'+str(foldlist[0])+"f_rep"+str(rep)+'_task_'+str(task_token)+".txt",'w')
    return dirt,logfile,datalist,metalist,timestamp,fitmetrics
#################################################################################
#################################################################################
#################################################################################
###             Inputs                                  #
#################################################################################
task ='it'
runlist =['4','5','6','7']
rep= 1 
foldlist = [0]

timelist =[100]
prep = False
dirt = '/root/data/'
task_token = secrets.token_hex(8)
#################################################################################
dirt,logfile,datalist,metalist,timestamp,fitmetrics = init(dirt,task,runlist,timelist,foldlist,rep,task_token)
if debugmode:
  pass
else:
  sys.stdout = logfile
#################################################################################
##########
#################################################################################
print(datalist,metalist)
for im,meta in enumerate(metalist):
    myid = meta.split('_')[0]
    if myid[2:] in runlist:
      print(myid[2:])
      framework = 'autosklearn'
      ncore = 4
      dataset = datalist[im]# "uci_bank_marketing_pd"
      print("\ndataset:\t",dataset)
      print("\nmetadata information:\t",meta)
      try:
        runbenchmark(prep,dataset,framework,foldlist,ncore,timelist,dirt,meta,fitmetrics,rep,logfile,task_token)
      except:
        traceback.print_exc(file=sys.stdout)
        print('Failed:\t',myid,dataset)
        continue
#################################################################################
##########
#################################################################################
for lf,locfold in enumerate(timelist):
    ldirt='results/'+str(locfold)+'s/'
    dataname,auclist,loglosslist,acclist=get_results_reg(ldirt,task_token)
#################################################################################
#################################################################################
##########
#################################################################################
if debugmode:
  pass
else:   
  sys.stdout = orig_stdout
  logfile.close()
