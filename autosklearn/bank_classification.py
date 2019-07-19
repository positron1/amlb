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


data = pd.read_csv(dirt+dataset+".csv")

data_train = dtrain.append(dvalid)

dtest = dtest.drop(['_dmIndex_','_PartInd_'],axis=1)
data_train = data_train.drop(['_dmIndex_','_PartInd_'],axis=1)

column= data_train.columns.values
X_test = dtest[column[:-1]]
y_test = dtest[column[-1]]
##################################################

print(column)
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\
    ('scaler', StandardScaler())])

numeric_features = ['age','duration','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_features = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month','day_of_week', 'campaign', 'poutcome']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
    ('onehot', OneHotEncoder())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
     ('cat', categorical_transformer, categorical_features)])
lb = preprocessing.LabelBinarizer()
################################################################################
X = pd.DataFrame(data.drop(['y'],axis=1))
data["y"]=data.where(data["y"]=='yes',1)
data["y"]=data.where(data["y"]=='no',0)
#for cf in categorical_features:
#   labels,uniques= pd.factorize(data[cf])
#   data[cf]=labels
#
#print(data[categorical_features])
##y = lb.fit_transform(y)

#y_test = lb.fit_transform(y_test)
print("before",X)

#X_train = preprocessor.fit_transform(X_train)
X = preprocessor.fit_transform(X)
print("after,",X)
print(OneHotEncoder.feature_indices_)
print("\n\n")

X_train =X["_PartInd_"]>0

X_test =X["_PartInd_"]==0
X_test = X_train.drop(['_dmIndex_','_PartInd_'],axis=1)
X_train = X_train.drop(['_dmIndex_','_PartInd_'],axis=1)
y = data['y','_dmIndex_','_PartInd_']
print(y)

#######################################################################################
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeforjob,\
        delete_tmp_folder_after_terminate=False,\
        n_jobs=4,\
        resampling_strategy='cv',\
        resampling_strategy_arguments={'folds': int(foldn)},)

##########################################

#clf = Pipeline(steps=[('preprocessor', preprocessor),
print(np.ndim(X_train[0]))
automl.fit(X_train, y_train)
print(np.ndim(X_train[0]))
automl.refit(X_train, y_train)
print(np.ndim(X_train[0]))
print(np.ndim(X_test[0]))
y_pred = automl.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
print(automl.sprint_statistics())
print(automl.show_models())
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
print(type(automl.show_models()))
#print(automl.cv_results_)
print("automl.fit")
print(automl.fit(X_test, y_test))
finalmodel_file ='finalmodelemsenble.pkl'
finalmodel = open(finalmodel_file,'wb')
pickle.dump((automl,X_train,y_train),finalmodel)
finalmodel.close()

#save_object(automl.cv_results_,str("cv_results")+resultfile)
