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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score,accuracy_score
##################################################
import pickle
from DateTime import DateTime
import time

current_time = DateTime(time.time(), 'US/Eastern')
framework = 'autosklearn'
datasetn = 'kaggle_santander_pd'
foldn =  '3'
timeforjob=360
dirt = '/root/data/'
ncore = 8 
prepart = True
resultfile = str(datasetn)+str(foldn) +"fold"+ str(timeforjob) + "seconds" + str(ncore)+"core"+\
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)+'prepart.txt'

dataset = 'kaggle_santander_pd'
dtrain = pd.read_csv(dirt+dataset+"train.csv")
dvalid =pd.read_csv(dirt+dataset+"valid.csv")
dtest =pd.read_csv(dirt+dataset+"test.csv")

ctrain=  dtrain.append(dvalid)
data = pd.read_csv(dirt+dataset+".csv")

#data_train = dtrain.append(dvalid)
#dtest = dtest.drop(['_dmIndex_','_PartInd_'],axis=1)
#data_train = data_train.drop(['_dmIndex_','_PartInd_'],axis=1)

column= data.columns.values
index = ['_dmIndex_','_PartInd_']
print(index)
print(data[column[-2:]])
##################################################
#numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
#categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week', 'campaign', 'poutcome']
#data["y"]=data.where(data["y"]=='yes',1)
#data["y"]=data.where(data["y"]=='no',0)
################################################################################
X = data.drop(['target'],axis=1)

y = data[index+['target']]

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

briefout = open('prepart_result.csv','a')
briefout.write("dataset\t"+"fold\t"+"timelimit(second)\t"+"core\t"+"prepartitioned\t"+"normalized\t"+"ACC\t"+"AUC\t"+"log_loss\n")
briefout.write(str(datasetn)+"\t"+str(foldn) +"\t"+str(timeforjob)+"\t"+ str(ncore)+"\t"+str(prepart)+"\t"+str('True')+"\t"+str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\t"+str(roc_auc_score(y_test, y_pred))+"\t"+str(log_loss(y_test, y_pred))+"\n")
briefout.close()
##############################################################
resultfileout = open(resultfile,'w')
resultfileout.write(str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\n")
resultfileout.write(str(automl.show_models()))
resultfileout.write(str(automl.sprint_statistics()))
resultfileout.write(str(automl.cv_results_))
resultfileout.write(str(y_pred)+"\n"+str(y_test))
resultfileout.close()


