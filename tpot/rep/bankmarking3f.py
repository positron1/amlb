########### This is replcate of TPOT model 
## For dataset: bank marketing
## with 3 fold
## and other parameter as below

'''
{'fitness': '(1.0, 0.9327922582893441)',
 'model': 'XGBClassifier(input_matrix, XGBClassifier__learning_rate=0.1, '
          'XGBClassifier__max_depth=7, XGBClassifier__min_child_weight=6, '
          'XGBClassifier__n_estimators=100, XGBClassifier__nthread=1, '
          'XGBClassifier__subsample=0.6000000000000001)',
 'pipeline': Pipeline(memory=None,
     steps=[('xgbclassifier', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=7, min_child_weight=6, missing=None,
       n_estimators=100, n_jobs=1, nthread=1, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=None, subsample=0.6000000000000001, verbosity=1))])}i

id,task,framework,fold,result,mode,version,params,tag,utc,duration,models,seed,info,acc,auc
openml.org/t/14965,bank-marketing,TPOT,3,0.934107,local,0.9.6,,stable,2019-07-08T17:32:54,7900.8,469,3803467872,,0.911303,0.934107
'''




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import xgboost as xgb
import sklearn
###### Read in data
import pandas as pd
import numpy as np
import pickle
from DateTime import DateTime
import time

current_time = DateTime(time.time(), 'US/Eastern')
framework = 'tpot'
datasetn = 'bankmarketing'
foldn =  '3'
dirt = '../'

resultfile = str(framework) + str(datasetn) + str(foldn) + \
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())+'.txt'



data =pd.read_csv("/home/test/bank.csv",delimiter=';')
column= data.columns.values
X = data[column[:-1]]
y = data[column[-1]]
lb = preprocessing.LabelBinarizer()
y= lb.fit_transform(y)
##################################################
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    ('scaler', StandardScaler())])
numeric_features = ['age','balance','day','duration','pdays','previous']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'campaign', 'poutcome']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
     ('cat', categorical_transformer, categorical_features)])
################################################################################
X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

xgb.fit(X_train, y_train, eval_metric='auc')  # works fine

# Making a pipeline with this classifier and a scaler:
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', XGBClassifier())])

# using the pipeline, but not optimizing for 'auc': 
pipe.fit(X_train, y_train)  # works fine

# however this does not work (even after correcting the underscores): 
print(pipe.fit(X_train, y_train, classifier__eval_metric='auc'))  # fails
from sklearn.metrics import accuracy_score 

y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
