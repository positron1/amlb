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
from xgboost import XGBClassifier
import xgboost
import sklearn

print('sklearn version: %s' % sklearn.__version__)
print('xgboost version: %s' % xgboost.__version__)

X, y = load_iris(return_X_y=True)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=143)
# Without using the pipeline: 
xgb = XGBClassifier()
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
