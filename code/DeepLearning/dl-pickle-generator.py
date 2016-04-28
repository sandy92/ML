# uses https://github.com/lisa-lab/deeplearningtutorials
import pickle
import gzip
import pandas as pd
import numpy as np
from sklearn import cross_validation

import os
os.chdir("/Users/sandy/work/UF/ML/Project/code/DeepLearning")

""" Digits dataset """
""" optdigits.tra """
train = pd.read_csv("../../datasets/optdigits/optdigits.tra",header=None)
num_columns = len(train.columns)
features = train.columns[:-1]
X_train = train[features]
y_train = train[num_columns-1]

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=0)

""" optdigits.tes """
test = pd.read_csv("../../datasets/optdigits/optdigits.tes",header=None)
num_columns = len(train.columns)
features = test.columns[:-1]
X_test = test[features]
y_test = test[num_columns-1]

data = ((X_train.as_matrix(),y_train.as_matrix()),(X_valid.as_matrix(),y_valid.as_matrix()),(X_test.as_matrix(),y_test.as_matrix()))

f = gzip.open("data/digits.pkl.gz","wb")
pickle.dump(data,f)
f.close()

""" Breast cancer dataset """
""" breast-cancer-wisconsin.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/breast-cancer-wisconsin.data",header=None)
data = data.replace('?',np.nan).astype(float,raise_on_error=False).dropna(how='any') # removing missing values
num_columns = len(data.columns)
features = data.columns[1:-1]
X = data[features]
y = data[num_columns-1]/2 - 1
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=0)

data = ((X_train.as_matrix(),y_train.as_matrix()),(X_valid.as_matrix(),y_valid.as_matrix()),(X_test.as_matrix(),y_test.as_matrix()))
f = gzip.open("data/breast-cancer-wisconsin.pkl.gz","wb")
pickle.dump(data,f)
f.close()

""" wdbc.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/wdbc.data",header=None)
data = data.astype(float,raise_on_error=False).dropna(how='any') # removing missing values
features = data.columns[2:]
X = data[features]
y = data[1].replace(['M','B'],[0,1])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=0)

data = ((X_train.as_matrix(),y_train.as_matrix()),(X_valid.as_matrix(),y_valid.as_matrix()),(X_test.as_matrix(),y_test.as_matrix()))
f = gzip.open("data/wdbc.pkl.gz","wb")
pickle.dump(data,f)
f.close()

""" wpbc.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/wpbc.data",header=None)
data = data.replace('?',np.nan).astype(float,raise_on_error=False).dropna(how='any') # removing missing values
features = data.columns[3:]
X = data[features]
y = data[1].replace(['R','N'],[0,1])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=0)

data = ((X_train.as_matrix(),y_train.as_matrix()),(X_valid.as_matrix(),y_valid.as_matrix()),(X_test.as_matrix(),y_test.as_matrix()))
f = gzip.open("data/wpbc.pkl.gz","wb")
pickle.dump(data,f)
f.close()

""" Forest data set """
""" training.csv """
train = pd.read_csv("../../datasets/ForestTypes/training.csv")
features = train.columns[1:]
X_train = train[features]
X_train = X_train.astype(float)
y_train = train['class'].map(str.strip).replace(['d','h','o','s'],[0,1,2,3])

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=0)

""" testing.csv """
test = pd.read_csv("../../datasets/ForestTypes/testing.csv")
features = test.columns[1:]
X_test = test[features]
X_test = X_test.astype(float)
y_test = test['class'].map(str.strip).replace(['d','h','o','s'],[0,1,2,3])

data = ((X_train.as_matrix(),y_train.as_matrix()),(X_valid.as_matrix(),y_valid.as_matrix()),(X_test.as_matrix(),y_test.as_matrix()))
f = gzip.open("data/forest-types.pkl.gz","wb")
pickle.dump(data,f)
f.close()

with gzip.open('/Users/sandy/work/UF/ML/Project/code/DeepLearning/data/wpbc.pkl.gz', 'rb') as f:
    a = pickle.load(f)
    train_set, valid_set, test_set = a
    train_set_x,train_set_y = train_set
    test_set_x,test_set_y = test_set
    valid_set_x,valid_set_y = valid_set
print len(train_set_x[1]),len(train_set_x),len(train_set_y), len(valid_set_x),len(valid_set_y), len(test_set_x),len(test_set_y)
