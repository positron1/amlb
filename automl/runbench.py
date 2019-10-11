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

from DateTime import DateTime
import time
import time


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def getfitmetrics(fitmetrics):
    if fitmetrics == autosklearn.metrics.roc_auc:
       return 'AUC'
    elif fitmetrics == autosklearn.metrics.log_loss:
       return 'LogLoss'
def savemodel(timeforjob,resultfile,automl):
    resultfileout = open('results/'+str(timeforjob)+'s/finalmodels'+resultfile,'w')
    resultfileout.write(str(automl.show_models()))
    #resultfileout.write(str(automl.sprint_statistics()))
    resultfileout.write(str(automl.cv_results_))
    resultfileout.close()
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
def autoclf(framework,timeforjob,foldn,ncore,X_train,y_train,fitmetrics):
    if foldn ==0:
        
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
           per_run_time_limit=timeforjob, \
           delete_tmp_folder_after_terminate=False,\
           seed=1,\
           ensemble_memory_limit=20720,\
           resampling_strategy='holdout',\
           ml_memory_limit=20720*2,\
           resampling_strategy_arguments={'train_size': float(5/7)},
           n_jobs=ncore)
        automl.fit(X_train.copy(), y_train.copy(),metric=fitmetrics)
        automl.refit(X_train.copy(), y_train.copy())#,metric=autosklearn.metrics.roc_auc)
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
           delete_tmp_folder_after_terminate=False,\
           seed=1,\
           per_run_time_limit=timeforjob, \
           ensemble_memory_limit=20720,\
           ml_memory_limit=20720*2,\
           resampling_strategy_arguments={'folds': int(foldn)},
           resampling_strategy='cv',
           n_jobs=ncore) 
        automl.fit(X_train.copy(), y_train.copy(),metric=fitmetrics)
        automl.refit(X_train.copy(), y_train.copy())#,metric=autosklearn.metrics.roc_auc)
    return automl

def get_run_info(automl,dataset,shape,timeforjob,ncore,foldn,framework,resultsfile,fitmetrics,metrics,timespend):
    runs = dict()
    runs['data']=str(dataset)
    runs['shape']=dict()
    runs['shape']['xtrain']=shape[0]
    runs['shape']['ytrain']=shape[1]
    runs['shape']['xtest']=shape[2]
    runs['shape']['ytest']=shape[3]
    runs['para']=dict()
    runs['para']['time']=timeforjob
    runs['para']['fitmetrics']=str(fitmetrics)
    runs['para']['refitmetrics']='def'
    runs['para']['cores']=str(ncore)
    runs['para']['folds']=str(foldn)
    runs['para']['framework']=str(framework)
    runs['timespend'] = timespend
    runs['results']=dict(metrics)
    print(runs)
    #jsonf = json.dumps(jsonpickle.encode(runs))
    jsonf = json.dumps(runs)
    f = open('results/'+str(timeforjob)+'s/result_'+str(getfitmetrics(fitmetrics))+resultsfile+".json","w")
    savemodel(timeforjob,resultsfile,automl)
    f.write(jsonf)
    f.close()
def save_prob(timeforjob,dataset,resultsfile,foldn,y_pred,y_pred_prob):
    briefout = open('results/'+str(timeforjob)+'s/'+dataset+resultsfile+str(foldn)+'fresult.csv','w')
    briefout.write("#ypred\typred_prob\n")
    for i,y in enumerate(y_pred):
       briefout.write(str(y)+'\t'+str(y_pred_prob[i])+'\n')
    briefout.close() 

def biclassifier(resultsfile,X_train, y_train,X_test, y_test,dataset,framework,foldn,ncore,timeforjob,dirt,meta,fitmetrics):

    shape = []
    shape = [X_train.shape, y_train.shape,X_test.shape, y_test.shape] 
    start = time.time()
    automl = autoclf(framework,timeforjob,foldn,ncore,X_train,y_train,fitmetrics) 
    ###################################################################
    y_pred = automl.predict(X_test)
    y_pred_prob = automl.predict_proba(X_test)
    end = time.time()
    timespend =float(end - start)
    save_prob(timeforjob,dataset,resultsfile,foldn,y_pred,y_pred_prob)
    metrics = metric(y_test,y_pred,y_pred_prob)
    get_run_info(automl,dataset,shape,timeforjob,ncore,foldn,framework,resultsfile,fitmetrics,metrics,timespend)


def runbenchmark(dataset,framework,foldlist,ncore,timelist,dirt,meta,fitmetrics,rep,logfile):
    mylist = dataset.split("_")
    myid = mylist[0]
    if not os.path.exists(dirt+'opentest/'+dataset):
        load_partition(dirt+'opentest/',dataset)
    try:
        if os.path.exists(dirt+"meta/"+meta):
            nfeatures,cfeatures,target = meta_info(dirt,meta)
        else:
            nfeatures,cfeatures,target = autoprep(dirt,dataset,targetname) 

        data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False)
        print(type(X_train),X_train, y_train,X_test, y_test)
        for timeforjob in timelist:
            for foldn in foldlist:
                for _ in range(rep):
                    current_time = DateTime(time.time(), 'US/Eastern')
                    resultsfile = myid+"_"+str(framework)+'_'+str(foldn)+'f_'+str(ncore)+"c_"+str(timeforjob)+"s_"+str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
                    str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] 
                    print("\nstarting:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n',file=logfile)
                    biclassifier(resultsfile,X_train, y_train,X_test, y_test,dataset,framework,foldn,ncore,timeforjob,dirt,meta,fitmetrics)
                    print("Finishing:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
    except:
        print("\nfail in:\t",dataset)
        traceback.print_exc(file=sys.stdout)


