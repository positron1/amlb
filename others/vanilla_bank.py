## Yonglin Zhu
##
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score,accuracy_score##################################################
import pickle
from DateTime import DateTime
import time
import sys
import coremltools
import sklearn
from xgboost import XGBClassifier
import xgboost
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
current_time = DateTime(time.time(), 'US/Eastern')
###################################################################
# Use sklearn to holdout, read in original data
#######################################################################

framework = 'autosklearn'
datasetn = 'bankmarketing'
foldn =  '3'
timeforjob = 3600
ncore = 4
prepart = False
dirt = '/root/data/'
############################################################################################################
resultfile = str(datasetn)+str(foldn) +"fold"+ str(timeforjob) + "seconds" + str(ncore)+"core"+\
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)+'.txt'


#dataset = "uci_bank_marketing_pd"
#data = pd.read_csv(dirt+dataset+".csv") # panda.DataFrame
data =pd.read_csv("/home/test/bank.csv",delimiter=';')
print(data.columns)
#numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
numeric_features = ['age','balance','day','duration','pdays','previous']
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'campaign', 'poutcome']

#categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week', 'campaign', 'poutcome']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
    ('onehot', OneHotEncoder(sparse=False))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
     ('cat', categorical_transformer, categorical_features)])

######################################################################
X = data[categorical_features+numeric_features]
y= data["y"]
lb = preprocessing.LabelBinarizer()
y= lb.fit_transform(y)
##########################################################
##################################################################
X=preprocessor.fit_transform(X)
#pd.DataFrame(X).to_csv('X_vanilla.csv')
X_train, X_test, y_train, y_test = \
  sklearn.model_selection.train_test_split(X, y,test_size=0.2, random_state=1)
#pd.DataFrame(X_train).to_csv("xtrain_vanilla.csv")
#pd.DataFrame(y_train).to_csv("ytrain_vanilla.csv")
print(X_train)
#################################################################################


xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,\
       max_delta_step=0, max_depth=7, min_child_weight=6, missing=None,\
       n_estimators=100, n_jobs=1, nthread=1, objective='binary:logistic',\
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,\
       seed=None, silent=None, subsample=0.6000000000000001, verbosity=1)
#xgb.fit(X_train,y_train, eval_metric='auc')  # works fine

###########################################################################
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
        per_run_time_limit=int(timeforjob/10),\
        delete_tmp_folder_after_terminate=False,\
        ensemble_memory_limit=10240,
        seed=1,
        ml_memory_limit=30720,
        n_jobs=ncore,\
        resampling_strategy_arguments={'folds': int(foldn)},
        resampling_strategy='cv',)
#automl.fit(X_train.copy(), y_train.copy())
#automl.refit(X_train.copy(),y_train.copy())
###################################################################
###################################################################
xgb.fit(X_train.copy(), y_train.copy())

###################################################################
###################################################################
y_pred0 = xgb.predict(X_test)
#y_pred = automl.predict(X_test)
######################################################################
briefout = open('vanilla_result.csv','a')
briefout.write("dataset\t"+"fold\t"+"timelimit(second)\t"+"core\t"+"prepartitioned\t"+"ACC\t"+"AUC\t"+"log_loss\n")
briefout.write(str(datasetn)+","+str(foldn) +","+str(timeforjob)+","+ str(ncore)+","+str(prepart)+","+str(sklearn.metrics.accuracy_score(y_test, y_pred0))+","+str(roc_auc_score(y_test, y_pred0))+","+str(log_loss(y_test, y_pred0))+"\n")
#briefout.write(str(datasetn)+","+str(foldn) +","+str(timeforjob)+","+ str(ncore)+","+str(prepart)+","+str(sklearn.metrics.accuracy_score(y_test, y_pred))+","+str(roc_auc_score(y_test, y_pred))+","+str(log_loss(y_test, y_pred))+"\n")
briefout.close()
##############################################################
#resultfileout = open(resultfile,'w')
#resultfileout.write(str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\n")
#resultfileout.write(str(automl.show_models()))
#resultfileout.write(str(automl.sprint_statistics()))
#resultfileout.write(str(automl.cv_results_))
#resultfileout.write(str(y_pred)+"\n"+str(y_test))
#resultfileout.close()

