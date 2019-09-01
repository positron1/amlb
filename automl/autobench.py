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

from sas7bdat import SAS7BDAT
import pandas as pd
import os
import sys
import logging
import optparse
from utils import *

from DateTime import DateTime
import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
current_time = DateTime(time.time(), 'US/Eastern')


framework = 'autosklearn'
foldn = 5
timeforjob= 4*900
prepart = True
ncore = 4

def metric(y_test,y_pred,y_pred_prob):
    metrics = dict()
    metrics['logloss']=log_loss(y_test,y_pred_prob)
    metrics['AUC']=roc_auc_score(y_test,y_pred)
    metrics['f1']=f1_score(y_test,y_pred)
    metrics['ACC']=accuracy_score(y_test,y_pred)
    return metrics
def autoprep(dirt,dataset,targetname):
    if targetname:
        targetname = dataset
    else:
        dirt = dataset
        # use the last one as target and print it out
    return nfeatures,cfeatures,target
numeric_features =[]
categorical_features =[]
dirt = '../data/'
datalist = glob.glob(dirt+"opentest/*sas7bdat")
metalist = glob.glob(dirt+"meta/*csv")
datalist = remove_dirt(datalist,dirt+'/opentest/')
metalist = remove_dirt(metalist,dirt+'/meta/')
for im,meta in enumerate(metalist):
    resultsfile = str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
    str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)
    runs = dict()
    dataset = datalist[im]# "uci_bank_marketing_pd"
    print("\ndataset:\t",dataset)
    print("\nmetadata information:\t",meta)
    if not os.path.exists(dirt+'opentest/'+dataset):
        load_partition(dirt+'opentest/',dataset)
    runs['data']=dataset
    try:
        if os.path.exists(dirt+"meta/"+meta):
            nfeatures,cfeatures,target = meta_info(dirt,meta)
        else:
            nfeatures,cfeatures,target = autoprep(dirt,dataset,targetname) 

        data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False)
        print("\nstarting:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
                delete_tmp_folder_after_terminate=False,\
                seed=1,\
                ensemble_memory_limit=10240,\
                ml_memory_limit=30720,\
                n_jobs=ncore)
        
        automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
        automl.refit(X_train.copy(),y_train.copy())
        ###################################################################
        runs['para']=dict()
        runs['para']['time']=timeforjob
        runs['para']['cores']=ncore
        runs['para']['folds']=foldn
        runs['para']['framework']=framework
        y_pred = automl.predict(X_test)
        y_pred_prob = automl.predict_proba(X_test)
        briefout = open('results/'+str(timeforjob)+'s/'+dataset+resultsfile+str(foldn)+'fresult.csv','w')
        briefout.write("#ypred\typred_prob\n")
        for i,y in enumerate(y_pred):
           briefout.write(str(y)+'\t'+str(y_pred_prob[i])+'\n')
        briefout.close() 
        print("Finishing:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        print(y_pred) 
        print(y_pred_prob)
        metrics = metric(y_test,y_pred,y_pred_prob)
        runs['results']=metrics
        jsonf = json.dumps(runs)
        f = open('results/'+str(timeforjob)+'s/'+dataset+resultsfile+".json","w")
        f.write(jsonf)
        f.close()
    except:
        print("\nfail in:\t",dataset)
        traceback.print_exc(file=sys.stdout)
