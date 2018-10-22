#!/usr/bin/env python

"""
Some code to generate figures for slides.
"""

## Basic packages
import numpy as np
import pandas as pd 
from sklearn.externals.six import StringIO 
import warnings
warnings.filterwarnings('ignore')

## xgboost
import xgboost as xgb
from xgboost_helpers import *

## Ploting 
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

## sklearn for model selection 
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

## sklearn logistic regression and random forest classifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

set_mpl_params()

## Load data 
df = pd.read_csv("../data/motion_smartphone.csv")
df = df.drop("subject", axis=1)

## Convert the activity labels from 1-6 to 0-5
df.activity -= 1

## Generate training and testing set, including all predictor variables  
x_train, x_test, y_train, y_test =\
    train_test_split(df.drop(["activity"], axis = 1),
                     df["activity"], test_size = 0.3,
                     random_state = 5)

## Load the data into xgb data format
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

## -------------------------------------------------------------------
### Train a xgboost model
bparam = {'max_depth': 6,                      # max. tree depth
         'eta': 0.3,                           # learning rate
         'silent': 1,                          # 0 to print messages
         'objective': 'multi:softmax',         # multi-class softmax
         'num_class': 6,                       # classes for softmax
         'nthread': 4,                         # threads to use
         'eval_metric': ['merror', 'mlogloss'] # no 'auc' for multi-class
}

## Validations set to watch performance
evallist = [(dtest, 'test'), (dtrain, 'train')]
evals = dict()                  # store validations at each iteration

# xgboost on a training set
num_round = 50
mod = xgb.train(bparam, dtrain, num_round, evals=evallist,
                evals_result=evals, verbose_eval=False)

# Show how validation functions have major effect on test performance
plot_xgb_evals(evals, n_plot=2, label=True)

    
## -------------------------------------------------------------------
### Predictions
preds = mod.predict(dtest)
cm, report = xgb_metrics(mod, dtest, y_test)
bparam['eval_metric'] = 'merror'

# custom function to evaluate on the cv test-set each iteration of cross-validation
res = xgb.cv(bparam, xgb.DMatrix(df.drop('activity', axis=1), df.activity),
             nfold=30, feval=xgb_eval_accuracy)

## -------------------------------------------------------------------
### TODO: Show GB doesn't overfit relative to other classifiers

dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=1)
train_sizes, train_scores, valid_scores =\
    learning_curve(dt_clf, df.drop('activity', axis=1), df.activity, cv=5)
