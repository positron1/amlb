import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
###### Read in data
import pandas as pd
import numpy as np

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
def readin(pathtodata):
    data =pd.read_csv(pathtodata,delimiter=';')
    column= data.columns.values
    X = data[column[:-1]]
    y = data[column[-1]]
    lb = preprocessing.LabelBinarizer()
    y= lb.fit_transform(y)
    return X,y
##################################################

def preprocess(nimputer,nscaler,numeric_features,cimputer,categorical_features,encoder):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
        ('scaler', StandardScaler())])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
         ('cat', categorical_transformer, categorical_features)])
    return preprocessor
################################################################################
def getpara():
   ############## read from show_model 
pathtodata = "/home/test/bank.csv"
numeric_features = ['age','balance','day','duration','pdays','previous']
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'campaign', 'poutcome']
#### parameter of final models #######
nimputer = ['median']
nscaler = ['scaler']
cimputer = ['constant']
cscaler = ['']
encoder = ['onehot']
X,y = readin(pathtodata)
for i,n in enumerate(nimputer):
   preprocessor = preprocess(nimputer[i],nscaler[i],numeric_features,cimputer[i],categorical_features,encoder[])
   clf = RandomForestClassifier(

#
#automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=72,\
#        delete_tmp_folder_after_terminate=False,\
#        resampling_strategy='cv',\
#        resampling_strategy_arguments={'folds': int(foldn)},)
   clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', automl)])
X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)

clf.fit(X_train, y_train)

#pattern = r"(?P<framework>[\w\-]+?)_(?P<task>[\w\-]+)_(?P<fold>\d+)(_(?P<datetime>\d{8}T\d{6}))?.csv"

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
# sample usage
save_object(automl.show_models(),str("showmodels")+resultfile)
print(automl.cv_results_)
print("clf.fit")
print(clf.fit(X_test, y_test))
#save_object(automl.cv_results_,str("cv_results")+resultfile)
