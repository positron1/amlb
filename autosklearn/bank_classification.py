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
data =pd.read_csv("/home/test/bank.csv",delimiter=';')

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


column= data.columns.values
X = data[column[:-1]]
y = data[column[-1]]
lb = preprocessing.LabelBinarizer()
y= lb.fit_transform(y)
print(y[:50])
automl = autosklearn.classification.AutoSklearnClassifier()

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', automl)])

X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

print(automl.show_models())
#automl.fit(X_train, y_train)
#y_hat = automl.predict(X_test)
#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
