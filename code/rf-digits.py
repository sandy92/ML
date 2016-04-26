import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import timeit
import os
os.chdir("/Users/sandy/work/UF/ML/Project/code")

train = pd.read_csv("../datasets/optdigits/optdigits.tra",header=None)
features = train.columns[:-1]
X_train = train[features]
y_train = train[64]

test = pd.read_csv("../datasets/optdigits/optdigits.tes",header=None)
features = test.columns[:-1]
X_test = test[features]
y_test = test[64]

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print "random forest accuracy: ",acc_rf

# print train.head()
# print type(train)

# train = pd.read_csv("train.csv")
# features = train.columns[1:]
# X = train[features]
# y = train['label']
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X/255.,y,test_size=0.1,random_state=0)
#
# clf_rf = RandomForestClassifier()
# clf_rf.fit(X_train, y_train)
# y_pred_rf = clf_rf.predict(X_test)
# acc_rf = accuracy_score(y_test, y_pred_rf)
# print "random forest accuracy: ",acc_rf
