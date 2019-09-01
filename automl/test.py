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
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import cross_val_score
##################################################

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
foldn = 3
timeforjob= 100
prepart = True
ncore = 4
dirt ='/home/yzhu14/atoml/data/fraudkaggle/'
dataset = 'ieeefraud'

def kaggledata(dataset,dirt,nfeature,cfeature,target,indexfrop):
  trainn = pd.read_csv()
  trainc = pd.read_csv()
  testn = pd.read_csv()
  testc = pd.read_csv()

  return  data,X,y,X_train, y_train,X_test, y_test
#######################################################
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
                per_run_time_limit=int(timeforjob/10),\
                delete_tmp_folder_after_terminate=False,\
                ensemble_memory_limit=10240,
                seed=1,
                ml_memory_limit=30720,
                ensemble_size=5,
                n_jobs=ncore)
        
        
        automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
        automl.refit(X_train.copy(),y_train.copy())
        ###################################################################
        y_pred = automl.predict(X_test)
        print("Finishing:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        briefout = open(framework+'result.csv','a')
        briefout.write("dataset\t"+"fold\t"+"timelimit(second)\t"+"core\t"+"prepartitioned\t"+"normalized\t"+"ACC\t"+"AUC\t"+
        "log_loss\n")
        briefout.write(str(dataset)+"\t"+str(foldn) +"\t"+str(timeforjob)+"\t"+ str(ncore)+"\t"+str(prepart)+"\t"+str('True'
        )+"\t"+str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\t"+str(roc_auc_score(y_test, y_pred))+"\t"+str(log_loss(
        y_test, y_pred))+"\n")
        briefout.close()
    except:
        print("\nfail in:\t",dataset)
        traceback.print_exc(file=sys.stdout)
