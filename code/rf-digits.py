""" random forest for handwritten digits dataset """
import pandas as pd
import numpy as np
from sklearn import cross_validation
from rf import rf # custom function (in rf.py) for random forests

import os
os.chdir("/Users/sandy/work/UF/ML/Project/code")

""" optdigits.tra """
train = pd.read_csv("../datasets/optdigits/optdigits.tra",header=None)
num_columns = len(train.columns)
features = train.columns[:-1]
X_train = train[features]
y_train = train[num_columns-1]

""" optdigits.tes """
test = pd.read_csv("../datasets/optdigits/optdigits.tes",header=None)
num_columns = len(train.columns)
features = test.columns[:-1]
X_test = test[features]
y_test = test[num_columns-1]

print rf(X_train, X_test, y_train, y_test,"Optical Recognition of Handwritten Digits")

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
