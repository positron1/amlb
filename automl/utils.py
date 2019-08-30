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
import os
import sys,traceback
import logging
import optparse

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def sas_to_csv(dirt,dataset):
    print("\n\nReading data from",dirt+ dataset)
    with SAS7BDAT(dirt + dataset) as f:
        df = f.to_data_frame()
    print("\n\nData description:\n\n",df.describe())
    cols = df.columns
    df.to_csv(dirt+dataset+'.csv',encoding = 'utf-8',index = False,header =True)
    print("\n\nCheck column\n\n",cols)
    return df

def load_partition(dirt,dataset):
    df = sas_to_csv(dirt,dataset)
    #### last column _PartInd_ for train-1/validation-2/test-0/
    cols = df.columns
    df._PartInd_.astype(int)
    dtrain = df.loc[df[cols[-1]]==1]
    dvalidate = df.loc[df[cols[-1]]==0]
    dtest = df.loc[df[cols[-1]]==2]
    print("Train\n",dtrain.shape)
    print("Validate\n",dvalidate.shape)
    print("Test\n",dtest.shape)
    return dtrain,dvalidate,dtest

def partition_to_csv(dirt,dataset,dtrain,dvalidate,dtest):
    dtrain,dvalidate,dtest = load_partition(dirt,dataset)
    dtrain.to_csv(dirt+dataset+'dtrain.csv',encoding = 'utf-8',index= False,header =True)
    dtest.to_csv(dirt+dataset+'dtest.csv',encoding = 'utf-8',index = False,header =True)
    dvalidate.to_csv(dirt+dataset+'dvalid.csv',encoding = 'utf-8',index=False,header =True)
def get_metrics(y_test,y_pred):
    metrics = dict()
     
def main(options,args):
    dirt = options.path
    dataset = options.data
    load_partition(dirt,dataset)

import glob

def prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False):
    index_features = ['_dmIndex_','_PartInd_']
    data = pd.read_csv(dirt+"opentest/"+dataset+'.csv',delimiter=delim) # panda.DataFrame
    data= data.astype({'_dmIndex_':'int', '_PartInd_':'int'})
    numeric_features = nfeatures #list(set(data.select_dtypes(include=["number"]))-set(index_features)-set([target]))
    categorical_features = cfeatures#list(set(data.select_dtypes(exclude=["number"]))-set(index_features)-set([target]))
    print("\nCheck numerical features:\t",numeric_features)
    print("\nCheck catogorical features:\t",categorical_features)
    ###############################
    index_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1))])
    y_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1)),\
                                   ('orden', OrdinalEncoder())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    #numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    #    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
        ('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
         ('cat', categorical_transformer, categorical_features), ('y',y_transformer,[target]),('index',index_transformer
, index_features)])

    data=preprocessor.fit_transform(data)
    data=pd.DataFrame(data)
    col =data.columns.values
    X=data.drop(col[-3:],axis=1)
    X_train = data[data[col[-1]]<2].drop(col[-3:],axis=1)  #pd.DataFrame(X).to_csv('X_vanilla.csv')
    X_test = data[data[col[-1]]==2].drop(col[-3:],axis=1)    #pd.DataFrame(X).to_csv('X_vanilla.csv')
    y=data[col[-3]]
    y_train =data[data[col[-1]]<2][col[-3]]
    y_test =data[data[col[-1]]==2][col[-3]]
    ##########################################################
    return data,X,y,X_train, y_train,X_test, y_test
def remove_dirt(dlist,dirt):
    for i,d in enumerate(sorted(dlist)):
        dlist[i]=os.path.relpath(d,dirt)
    return dlist
def meta_info(dirt,meta):
    dmeta = pd.read_csv(dirt+"meta/"+meta)
    target= dmeta[dmeta['ROLE']=='TARGET']
    targetname=target["NAME"].tolist()[0]
    inputs =  dmeta[dmeta['ROLE']=='INPUT']
    cinputs= inputs[inputs['type']=='C']
    cinputname = cinputs["NAME"].tolist()
    ninputs = inputs[inputs['type']=='N']
    ninputname = ninputs["NAME"].tolist()
    return ninputname,cinputname,targetname
if __name__ == '__main__':
    datadirt = '/root/data/'
    
    datalist = glob.glob(datadirt+"opentest/*sas7bdat")
    metalist = glob.glob(datadirt+"meta/*csv")
    framework = 'autosklearn'
    datasetn = 'bankmarketing'
    foldn = 0
    timeforjob= 3600
    prepart = True
    ncore = 8
    
    dirt = '/root/data/'
    numeric_features =[]
    categorical_features =[]
    datalist = remove_dirt(datalist,dirt+'/opentest/')
    metalist = remove_dirt(metalist,dirt+'/meta/')
    print(datalist,metalist)
    for im,meta in enumerate(metalist[3:]):
        print(meta)
        dataset = datalist[3:][im]# "uci_bank_marketing_pd"
        print(dataset)
        load_partition(dirt+'opentest/',dataset)
        nfeatures,cfeatures,target = meta_info(dirt,meta) 
        try:
            data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,nfeatures,cfeatures,target,delim=',',indexdrop=False)
            
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
                    per_run_time_limit=int(timeforjob),\
                    delete_tmp_folder_after_terminate=False,\
                    seed=1,
                    n_jobs=ncore)
            
            
            automl.fit(X_train.copy(), y_train.copy(),metric=autosklearn.metrics.roc_auc)
            automl.refit(X_train.copy(),y_train.copy())
            ###################################################################
            y_pred = automl.predict(X_test)
             
            briefout = dataset+framework+foldn+'result.csv'
            with open(briefout, 'w') as f:
                for item in y_pred:
                    f.write("%s\n" % item)
        except:
            print("\nfail in:\t",dataset)
            traceback.print_exc(file=sys.stdout)