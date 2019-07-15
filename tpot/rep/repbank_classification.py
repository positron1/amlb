import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
###### Read in data
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
##################################################
import pickle
from DateTime import DateTime
import time

current_time = DateTime(time.time(), 'US/Eastern')
framework = 'autosklearn'
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
X = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)

#######################################################################################
automl = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,\
       max_delta_step=0, max_depth=7, min_child_weight=6,\
       n_estimators=100, n_jobs=1, nthread=1, objective='binary:logistic',\
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,\
       subsample=0.6, verbosity=1)
##########################################

automl.fit(X_train, y_train)
y_pred = automl.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
# sample usage

print("automl.fit")
print(automl.fit(X_test, y_test))
finalmodel_file ='finalmodelemsenble.pkl'
finalmodel = open(finalmodel_file,'wb')
pickle.dump(automl,finalmodel)
finalmodel.close()

#save_object(automl.cv_results_,str("cv_results")+resultfile)
