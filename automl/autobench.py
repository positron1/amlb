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
from runbench import *

from DateTime import DateTime
import time

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
current_time = DateTime(time.time(), 'US/Eastern')


framework = 'autosklearn'
foldn = 5/7
timeforjob= 900
prepart = True
ncore = 4

numeric_features =[]
categorical_features =[]
dirt = '/root/data/'
datalist = glob.glob(dirt+"opentest/*sas7bdat")
metalist = glob.glob(dirt+"meta/*csv")
datalist = remove_dirt(datalist,dirt+'/opentest/')
metalist = remove_dirt(metalist,dirt+'/meta/')
print(datalist)
for im,meta in enumerate(metalist):
    runs = dict()
    dataset = datalist[im]# "uci_bank_marketing_pd"
    print("\ndataset:\t",dataset)
    print("\nmetadata information:\t",meta)
    runbenchmark(dataset,framework,foldn,ncore,timeforjob,dirt,meta)

#def autoclf(timeforjob,foldn,ncore,X_train,y_train):
#    print("\nstarting:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
#    if foldn ==0:
#        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
#               delete_tmp_folder_after_terminate=False,\
#               seed=1,\
#               resampling_strategy='holdout',
#               ml_memory_limit=100720,\
#               resampling_strategy_arguments={'train_size': float(5/7)},
#               n_jobs=ncore)
#               automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
#    else:
#        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
#               delete_tmp_folder_after_terminate=False,\
#               seed=1,\
#               ml_memory_limit=100720,\
#               resampling_strategy_arguments={'folds': int(foldn)},
#               resampling_strategy='cv',
#               n_jobs=ncore)o 
#               automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
#               automl.refit(X_train.copy(), y_train.copy())#,metric=autosklearn.metrics.roc_auc)
#    return automl
#
#def get_run_info(timeforjob,fitmetrics,ncore,foldn,framework,metrics,resultsfile):
#        runs['data']=dataset
#        runs['para']=dict()
#        runs['para']['time']=timeforjob
#        runs['para']['fitmetrics']='AUC'
#        runs['para']['refitmetrics']='def'
#        runs['para']['cores']=ncore
#        runs['para']['folds']=foldn
#        runs['para']['framework']=framework
#        runs['results']=metrics
#        jsonf = json.dumps(runs)
#        f = open('results/'+str(timeforjob)+'s/result_'+resultsfile+".json","w")
#        savemodel(resultsfile,automl)
#        f.write(jsonf)
#        f.close()
#def save_prob(timeforjob,dataset,resultsfile,foldn,y_pred):
#    briefout = open('results/'+str(timeforjob)+'s/'+dataset+resultsfile+str(foldn)+'fresult.csv','w')
#    briefout.write("#ypred\typred_prob\n")
#    for i,y in enumerate(y_pred):
#       briefout.write(str(y)+'\t'+str(y_pred_prob[i])+'\n')
#    briefout.close() 
#
#def runbenchmark(dataset,framework,foldn,ncore,timeforjob,dirt,meta):
#    resultsfile = dataset[:3]+"_"+str(framework)+'_'+str(foldn)+'f_'+str(ncore)+"c_"+str(timeforjob)+"s_"+str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
#    str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] 
#    if not os.path.exists(dirt+'opentest/'+dataset):
#        load_partition(dirt+'opentest/',dataset)
#    try:
#        if os.path.exists(dirt+"meta/"+meta):
#            nfeatures,cfeatures,target = meta_info(dirt,meta)
#        else:
#            nfeatures,cfeatures,target = autoprep(dirt,dataset,targetname) 
#
#        data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False)
#
#        automl = autoclf(timeforjob,foldn,ncore,X_train,y_train) 
#        ###################################################################
#        y_pred = automl.predict(X_test)
#        y_pred_prob = automl.predict_proba(X_test)
#        save_prob(timeforjob,dataset,resultsfile,foldn,y_pred):
#        print("Finishing:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
#        metrics = metric(y_test,y_pred,y_pred_prob)
#        get_run_info(timeforjob,fitmetrics,ncore,foldn,framework,metrics,resultsfile)
#    except:
#        print("\nfail in:\t",dataset)
#        traceback.print_exc(file=sys.stdout)
#
