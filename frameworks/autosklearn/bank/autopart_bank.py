## Yonglin Zhu
##
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
foldn =  3
timeforjob= 600*foldn 
prepart = True
ncore = 4
dirt = '/root/data/'
############################################################################################################
resultfile = str(datasetn)+str(foldn) +"fold"+ str(timeforjob) + "seconds" + str(ncore)+"core"+\
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)+'prepart.txt'
dataset = "uci_bank_marketing_pd"

numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed','campaign']
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week','poutcome']

def prep(dataset,dirt,numeric_features,categorical_features,delim=',',indexdrop=False):
    index_features = ['_dmIndex_','_PartInd_']          
    data = pd.read_csv(dirt+dataset+'.csv',delimiter=delim) # panda.DataFrame
    data= data.astype({'_dmIndex_':'int', '_PartInd_':'int'}) 
    ###############################
    index_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1))])
    y_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1)),\
                                   ('orden', OrdinalEncoder())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
        ('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
         ('cat', categorical_transformer, categorical_features), ('y',y_transformer,['y']),('index',index_transformer, index_features)])

    data=preprocessor.fit_transform(data)
    data=pd.DataFrame(data)
    col =data.columns.values
    X=data.drop(col[-3:],axis=1)
    X_train = data[data[col[-1]]>0].drop(col[-3:],axis=1)  #pd.DataFrame(X).to_csv('X_vanilla.csv')
    X_test = data[data[col[-1]]==0].drop(col[-3:],axis=1)    #pd.DataFrame(X).to_csv('X_vanilla.csv')

    ####################################################################
    y=data[col[-3]]
    y_train =data[data[col[-1]]>0][col[-3]]
    y_test =data[data[col[-1]]==0][col[-3]]
    ##################################################################
    
    X_train_auto, X_test_auto, y_train_auto, y_test_auto = \
      sklearn.model_selection.train_test_split(X, y,test_size=0.2, random_state=1)
    return data,X,y,X_train_auto, y_train_auto,X_test_auto, y_test_auto

data,X,y,X_train, y_train,X_test, y_test = prep(dataset,dirt,numeric_features,categorical_features,delim=',',indexdrop=False)
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

automl.fit(X_train.copy(), y_train.copy())
automl.refit(X_train.copy(),y_train.copy())
###################################################################
y_pred = automl.predict(X_test)
######################################################################
briefout = open('auto_part_result.csv','a')
briefout.write("dataset\t"+"fold\t"+"timelimit(second)\t"+"core\t"+"prepartitioned\t"+"normalized\t"+"ACC\t"+"AUC\t"+"log_loss\n")
briefout.write(str(datasetn)+"\t"+str(foldn) +"\t"+str(timeforjob)+"\t"+ str(ncore)+"\t"+str(prepart)+"\t"+str('True')+"\t"+str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\t"+str(roc_auc_score(y_test, y_pred))+"\t"+str(log_loss(y_test, y_pred))+"\n")
briefout.close()
##############################################################

resultfileout = open('auto_part'+resultfile,'w')
resultfileout.write(str(sklearn.metrics.accuracy_score(y_test, y_pred))+"\n")
resultfileout.write(str(automl.show_models()))
resultfileout.write(str(automl.sprint_statistics()))
resultfileout.write(str(automl.cv_results_))
resultfileout.write(str(y_pred)+"\n"+str(y_test))
resultfileout.close()


