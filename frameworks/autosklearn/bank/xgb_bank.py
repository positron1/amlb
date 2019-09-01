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
import lightgbm as lgb
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
foldn =  0
timeforjob= 1800 
prepart = True
ncore = 8
dirt = '/root/data/'
############################################################################################################
resultfile = str(datasetn)+str(foldn) +"fold"+ str(timeforjob) + "seconds" + str(ncore)+"core"+\
str(current_time.year()) + str(current_time.aMonth())+ str(current_time.day()) + \
str(current_time.h_24()) + str(current_time.minute())  + str(time.time())[:2] + str(framework)+'prepart.txt'
dataset = "uci_bank_marketing_pd"
numeric_features =[] 
categorical_features =[] 
def noprep(dataset,dirt,numeric_features,categorical_features,delim=',',indexdrop=False):
    index_features = ['_dmIndex_','_PartInd_']          
    data = pd.read_csv(dirt+dataset+'.csv',delimiter=delim) # panda.DataFrame
    print(data.columns)
    data= data.astype({'_dmIndex_':'int', '_PartInd_':'int'}) 
    numeric_features = list(set(data.select_dtypes(include=["number"]))-set(index_features)-set(['y']))
    categorical_features = list(set(data.select_dtypes(exclude=["number"]))-set(['y']))
    ###############################
    index_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1))])
    y_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-1)),\
                                   ('orden', OrdinalEncoder())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\
        ('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),\
         ('cat', categorical_transformer, categorical_features), ('y',y_transformer,['y']),('index',index_transformer, index_features)])

    ######################################################################
    data=preprocessor.fit_transform(data)
    data=pd.DataFrame(data)
    col =data.columns.values
    print(col)
    X=data.drop(col[-3:],axis=1)
    X_train = data[data[col[-1]]>0].drop(col[-3:],axis=1)  #pd.DataFrame(X).to_csv('X_vanilla.csv')
    X_test = data[data[col[-1]]==0].drop(col[-3:],axis=1)    #pd.DataFrame(X).to_csv('X_vanilla.csv')
    print(data.shape)

####################################################################
    #y= data["y"]
    #lb = preprocessing.LabelBinarizer()
    #y= lb.fit_transform(y)
    #data["y"]=data.where(data["y"]=='yes',1)
    #data["y"]=data.where(data["y"]=='no',0)
    y=data[col[-3]]
    y_train =data[data[col[-1]]>0][col[-3]]
    y_test =data[data[col[-1]]==0][col[-3]]
    ##########################################################
    ##################################################################
    feat_type = []#dict()
    xcol = X.columns.values
    for cl in xcol:
      if cl in  categorical_features:
        feat_type.append(1)
      else:
        feat_type.append(0)
    features = numeric_features+categorical_features          
    #X_train, X_test, y_train, y_test = \
      #sklearn.model_selection.train_test_split(X, y,test_size=0.2, random_state=1)
    return data,X,y,X_train, y_train,X_test, y_test,feat_type,features
data,X,y,X_train, y_train,X_test, y_test,categorical_indicator,features = noprep(dataset,dirt,numeric_features,categorical_features,delim=',',indexdrop=False)
#########################################################################################|\
model = lgb.LGBMClassifier(**{
     'learning_rate': 0.04,
     'num_leaves': 31,
     'max_bin': 1023,
     'min_child_samples': 10,
     'reg_alpha': 0.1,
     'reg_lambda': 0.2,
     'feature_fraction': 1.0,
     'bagging_freq': 50,
     'bagging_fraction': 0.85,
     'objective': 'binary',
     'n_jobs': -1,
     'n_estimators':200,})

from  sklearn.model_selection import StratifiedKFold
MODELS = []
train =X_train
target =y_train
oof = np.zeros(len(train))
predictions = np.zeros(len(X_test))
val_aucs = []
feature_importance_df = pd.DataFrame()
print(categorical_indicator)
catelist = []
for index,ci in enumerate(categorical_indicator):
   if ci ==1:
      catelist.append(index)
print(catelist)
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=11111)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4242)

param = {
    'bagging_freq': 5,  'bagging_fraction': 0.4,  'boost_from_average':'false',   
    'boost': 'gbdt',    'feature_fraction': 0.04, 'learning_rate': 0.01,
    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 10,
    'num_leaves': 13,  'num_threads': 8,            
    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': -1
}
features=list(set(features))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits



####################################################################################################1    
