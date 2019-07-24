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
##################################################
import pickle
from DateTime import DateTime
import time
import sys

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
timeforjob = 2*3600
ncore = 8
dirt = '/root/data/'
############################################################################################################
resultfile = str(datasetn)+str(foldn) +"fold"+ str(timeforjob) + "seconds" + str(ncore)+"core"+\
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)+'.txt'


dataset = "uci_bank_marketing_pd"
data = pd.read_csv(dirt+dataset+".csv") # panda.DataFrame

numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week', 'campaign', 'poutcome']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
    ('onehot', OneHotEncoder())])

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

X_train, X_test, y_train, y_test = \
  sklearn.model_selection.train_test_split(X, y,test_size=0.2, random_state=1)
pd.DataFrame(X_train).to_csv("xtrain_vanilla.csv")
pd.DataFrame(y_train).to_csv("ytrain_vanilla.csv")
print(X_train)
#################################################################################
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
        per_run_time_limit=int(timeforjob/10),\
        delete_tmp_folder_after_terminate=False,\
        ensemble_memory_limit=10240,
        seed=1,
        ml_memory_limit=30720,
        n_jobs=ncore,\
        resampling_strategy_arguments={'folds': int(foldn)},
        resampling_strategy='cv',)

#clf = Pipeline(steps=[('preprocessor', preprocessor),
#                      ('classifier', automl)])
automl.fit(X_train.copy(), y_train.copy())
automl.refit(X_train.copy(),y_train.copy())
###################################################################
y_pred = automl.predict(X_test)
######################################################################
briefout = open('vanilla_bank_result.csv','a')
briefout.write("dataset\t"+"fold\t"+"timelimit(second)\t"+"core\t"+"AUC\n")
briefout.write(str(datasetn)+","+str(foldn) +","+str(timeforjob)+","+ str(ncore)+","+str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\n")
briefout.close()
##############################################################
resultfileout = open(resultfile,'w')
resultfileout.write(str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\n")
resultfileout.write(str(automl.show_models()))
resultfileout.write(str(automl.sprint_statistics()))
resultfileout.write(str(automl.cv_results_))
resultfileout.close()


