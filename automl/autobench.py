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
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

framework = 'autosklearn'
foldn = 0
timeforjob= 3600
prepart = True
ncore = 8

def metric(y_test,y_pred,y_pred_prob):
    metrics = dict()
    metrics['logloss']=log_loss(y_test,y_pred_prob)
    metrics['AUC']=roc_auc_score(y_test,y_pred)
    metrics['f1']=f1_score(y_test,y_pred)
    metrics['ACC']=accuracy_score(y_test,y_pred)
    return metrics

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
datalist = glob.glob(dirt+"opentest/*sas7bdat")
metalist = glob.glob(dirt+"meta/*csv")
datalist = remove_dirt(datalist,dirt+'/opentest/')
metalist = remove_dirt(metalist,dirt+'/meta/')
for im,meta in enumerate(metalist):
    dataset = datalist[im]# "uci_bank_marketing_pd"
    print("\ndataset:\t",dataset)
    print("\nmetadata information:\t",meta)
    load_partition(dirt+'opentest/',dataset)
    try:
        nfeatures,cfeatures,target = meta_info(dirt,meta)
        data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False)
        print("\nstarting:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
                delete_tmp_folder_after_terminate=False,\
                seed=1,
                n_jobs=ncore)
        
        automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
        automl.refit(X_train.copy(),y_train.copy())
        ###################################################################
        y_pred = automl.predict(X_test)
        y_pred_prob = automl.predict_proba(X_test)
        briefout = dataset+framework+str(foldn)+'result.csv'
        with open(briefout, 'w') as f:
            for item in y_pred:
                f.write("%s\n" % item)
        f.close()
        print("Finishing:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        print(y_pred) 
        print(y_pred_prob)
        metrics = metric(y_test,y_pred,y_pred_prob)
        print(metrics)
        json = json.dumps(metrics)
        f = open(dataset+framework+str(foldn)+"metrics.json","w")
        f.write(json)
        f.close()
    except:
        print("\nfail in:\t",dataset)
        traceback.print_exc(file=sys.stdout)
