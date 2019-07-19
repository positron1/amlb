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
from sklearn.preprocessing import StandardScaler
from sksurv.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
##################################################
import pickle
from DateTime import DateTime
import time

current_time = DateTime(time.time(), 'US/Eastern')
framework = 'autosklearn'
datasetn = 'bankmarketing'
foldn =  '3'
timeforjob=360
dirt = '/root/data/'

resultfile = str(framework) + str(datasetn) + str(foldn) + \
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())+'.txt'
dataset = "uci_bank_marketing_pd"
dtrain = pd.read_csv(dirt+dataset+"train.csv")
dvalid =pd.read_csv(dirt+dataset+"valid.csv")
dtest =pd.read_csv(dirt+dataset+"test.csv")

ctrain=  dtrain.append(dvalid)
data = pd.read_csv(dirt+dataset+".csv")

#data_train = dtrain.append(dvalid)
#dtest = dtest.drop(['_dmIndex_','_PartInd_'],axis=1)
#data_train = data_train.drop(['_dmIndex_','_PartInd_'],axis=1)

column= data.columns.values
index = list(column[-2:])
print(index)
print(data[column[-2:]])
##################################################
numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week', 'campaign', 'poutcome']
data["y"]=data.where(data["y"]=='yes',1)
data["y"]=data.where(data["y"]=='no',0)
################################################################################
data_num = data[numeric_features]
data_num = data_num.fillna(data_num.mean())
data_num = pd.DataFrame(StandardScaler().fit_transform(data_num))
##### ONEHOTENCODING in scikit learn do not keep column names, so use pandas.get_dummies
data_cat = data[categorical_features].fillna('missing')
data_cat=pd.get_dummies(data_cat)

X = pd.concat([data[index],data_num,data_cat], axis=1)

y = data[index+['y']]

X_train =X[X['_PartInd_']>0]
X_test =X[X['_PartInd_']==0]
y_test =y[y['_PartInd_']==0]
y_train =y[y['_PartInd_']>0]

y_test = y_test.drop(columns=['_dmIndex_','_PartInd_']).astype(int)
y_train =y_train.drop(columns=['_dmIndex_','_PartInd_']).astype(int)
X_test = X_test.drop(columns=['_dmIndex_','_PartInd_'])
X_train =X_train.drop(columns=['_dmIndex_','_PartInd_'])



#######################################################################################
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
        per_run_time_limit=int(timeforjob/10),\
        delete_tmp_folder_after_terminate=False,\
        ensemble_memory_limit=10240,
        seed=1,
        ml_memory_limit=30720,
        n_jobs=8,\
        resampling_strategy='cv',\
        resampling_strategy_arguments={'folds': int(foldn)},)

##########################################
#clf = Pipeline(steps=[('preprocessor', preprocessor),
automl.fit(X_train, y_train)
automl.refit(X_train, y_train)
y_pred = automl.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
print(automl.sprint_statistics())
#pattern = r"(?P<framework>[\w\-]+?)_(?P<task>[\w\-]+)_(?P<fold>\d+)(_(?P<datetime>\d{8}T\d{6}))?.csv"

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
# sample usage
save_object(automl.show_models(),str("showmodels")+resultfile)
#kfold = KFold(n_splits=3, random_state=0)
#results = cross_val_score(automl, X_train, y_train, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("automodel1",automl.show_models())
print("automl.fit")
finalmodel_file ='finalmodelemsenble.pkl'
finalmodel = open(finalmodel_file,'wb')
pickle.dump((automl,X_train,y_train),finalmodel)
finalmodel.close()

print(automl.cv_results_)
