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
from collections import Counter 
def get_id(metalist):
  idlist=[]
  for im,meta in enumerate(metalist):
      myid = meta.split('_')[0]
      idlist.append(str(myid[2:]))
  return idlist
  
def common(str1,str2): 
      
    # convert both strings into counter dictionary 
    dict1 = Counter(str1) 
    dict2 = Counter(str2) 
  
    # take intersection of these dictionaries 
    commonDict = dict1 & dict2 
  
    if len(commonDict) == 0: 
        print -1
        return
  
    # get a list of common elements 
    commonChars = list(commonDict.elements()) 
  
    # sort list in ascending order to print resultant  
    # string on alphabetical order 
    commonChars = sorted(commonChars) 
   
    # join characters without space to produce  
    # resultant string 
    return ''.join(commonChars) 

logmode = False
if logmode: 
   orig_stdout = sys.stdout
current_time = DateTime(time.time(), 'US/Eastern')

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
outputdir = './results/'#'/run/user/yozhuz/automl/results/'
task ='bt'
if task=='bt':
   csvdatalist = glob.glob(dirt+"binaryTarget/data/*sas7bdat.csv")
   sasdatalist = glob.glob(dirt+"binaryTarget/data/*sas7bdat")
   metalist = glob.glob(dirt+"binaryTarget/meta/*csv")
   csvdatalist = remove_dirt(csvdatalist,dirt+'/binaryTarget/data/')
   sasdatalist = remove_dirt(sasdatalist,dirt+'/binaryTarget/data/')
   metalist = remove_dirt(metalist,dirt+'/binaryTarget/meta/')
   dirt = dirt+'binaryTarget/'
 #  outputdir = outputdir+'binaryTarget/'
elif task=='bre':
   sasdatalist = glob.glob(dirt+"binaryRareEvent/data/*sas7bdat")
   csvdatalist = glob.glob(dirt+"binaryRareEvent/data/*sas7bdat.csv")
   metalist = glob.glob(dirt+"binaryRareEvent/meta/*csv")
   csvdatalist = remove_dirt(csvdatalist,dirt+'/binaryRareEvent/data/')
   sasdatalist = remove_dirt(sasdatalist,dirt+'/binaryRareEvent/data/')
   metalist = remove_dirt(metalist,dirt+'/binaryRareEvent/meta/')
#   outputdir = outputdir+'binaryRareEvent/'
   dirt = dirt+'binaryRareEvent/'
print(csvdatalist)
print(sasdatalist)
print(metalist)
fitmetrics = autosklearn.metrics.log_loss
sasdatalist =sorted(sasdatalist)
csvdatalist =sorted(csvdatalist)
metalist = sorted(metalist)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
runlist =['0','1','2','3','4','10','11','12','13','14']
runlist =['12','13','14']
runlist =['5','6','7','8','9']
timelist = [900]
foldlist = [0]

prep = False
runlist = ['3']
runlist = ['4']

timestamp = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
        str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2]
runlist = ['9']
foldlist = [0]
runlist =['15','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
runlist =['0','1','2','3','4','5','6','7']
runlist =['0','1','2','3','4']
rep= 1
timelist = [3600]
rep=5
runlist =['15']
timelist = [900]
metalearning = True 
logfile = open('results/log_'+str(len(runlist))+'dataset'+str(timelist[0])+str(foldlist[0])+"rep"+str(rep)+str(timestamp)+".txt",'w')
if logmode:
  sys.stdout = logfile
corelist =[16]
print(runlist)
print("meta",metalist)
metadataid= get_id(metalist)
csvdataid = get_id(csvdatalist)
sasdataid = get_id(sasdatalist)
print('csvdataid',csvdataid)
print('sasdataid',sasdataid)
print(metadataid)
for ind in runlist:
    if ind in csvdataid: 
       dataset = csvdatalist[csvdataid.index(ind)]
    elif ind in sasdataid:
       dataset = sasdatalist[sasdataid.index(ind)]
       
    if ind in metadataid:
       meta = metalist[metadataid.index(ind)] 
    print(ind)
    framework = 'autosklearn'
    try:
      runbenchmark(metalearning,prep,dataset,framework,foldlist,corelist,timelist,dirt,meta,fitmetrics,rep,logfile,outputdir)
    except:
      print('Failed:\t',ind)#,dataset)
      traceback.print_exc(file=sys.stdout)
      continue
if logmode:
  sys.stdout = orig_stdout
  logfile.close()
