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
target = 'isFraud'
def kaggledata(dataset,dirt,nfeature,cfeature,target,indexdrop=False):
  trainn = pd.read_csv(dirt+'train_transaction.csv')
  trainc = pd.read_csv(dirt+'train_identity.csv')
  testn = pd.read_csv(dirt+'test_trainsaction.csv')
  testc = pd.read_csv(dirt+'test_identity.csv')
  nfeature = trainn.columns.values
  cfeature = trainc.columns.values
  train=trainc.merge(trainn)
  test = testc.merge(testc)
  data = pd.concat(train,test)
  index_features = ['TransactionID']
  index_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1))])
  y_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1)),\
                       ('orden', OrdinalEncoder())])
  numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
  categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
        ('onehot', OneHotEncoder(sparse=False))])
  preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
         ('cat', categorical_transformer, categorical_features), ('y',y_transformer,[target]),('index',index_transformer, index_features)])
  data=preprocessor.fit_transform(data)
  data=pd.DataFrame(data)
  y = data[target]
  X = data.drop([target],axis=1)
  X_train_auto, X_test_auto, y_train_auto, y_test_auto = \
      sklearn.model_selection.train_test_split(X, y,test_size=0.2, random_state=1)
  return  data,X,y,X_train, y_train,X_test, y_test
#######################################################
numeric_features =[]
categorical_features =[]
datalist = glob.glob(dirt+"opentest/*sas7bdat")
metalist = glob.glob(dirt+"meta/*csv")
datalist = remove_dirt(datalist,dirt+'/opentest/')
metalist = remove_dirt(metalist,dirt+'/meta/')
if __name__ == '__main__':
    try:
        data,X,y,X_train, y_train,X_test, y_test = kaggledata(dataset,dirt,nfeatures,cfeatures,target,indexdrop=False)
        print("\nstarting:\t",framework,'\t',foldn,' fold\t',ncore,' core\t', timeforjob,' seconds\n')
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
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
