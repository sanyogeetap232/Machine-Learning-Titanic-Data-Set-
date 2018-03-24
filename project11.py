# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 02:17:57 2017

@author: Sanyu
"""

# In[]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# In[]

train = pd.read_csv('C:\Anaconda\\train.csv')
train.head()

test = pd.read_csv('C:\Anaconda\\test.csv')
test.head()

train.describe()
test.describe()
# In[]
#imputing missing values with mean value for age

traincopy = train.fillna(train.mean())
traincopy.describe()

testcopy = test.fillna(test.mean())
testcopy.describe()

# In[]
testcopy = testcopy.drop(['Name'], axis = 1)

# In[]

testcopy.head()
# In[]

def f(row):
    if row['Sex'] == 'female':
        val = 1
    elif row['Sex'] == 'male':
        val =0
    return val

# In[]

traincopy['Sex'] = traincopy.apply(f, axis = 1)

# In[]

testcopy['Sex'] = testcopy.apply(f, axis = 1)
# In[]
testcopy.head()
# In[]

def e(row):
    if row['Embarked'] == 'C':
        val = 0
    elif row['Embarked'] == 'Q':
        val = 1
    else:
        val = 2
    return val

# In[]

traincopy['Embarked'] = traincopy.apply(e, axis = 1)
# In[]

testcopy['Embarked'] = testcopy.apply(e, axis = 1)

# In[]

#trainn = train.drop(['Name','Age', 'Ticket', 'Fare', 'Cabin'], axis = 1)

# In[]
X = traincopy[['Pclass', 'Sex', 'SibSp', 'Parch', 'Age']]
Y = traincopy['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
# In[]
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    pred = knn.predict(X_test)
    print ("Accuracy is", accuracy_score(Y_test, pred)*100, "for k = ", k)
    k = k + 1

# In[]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 8)

# In[]

knn.fit(X_train, Y_train)
knn.score(X_test, Y_test)

# In[]

pred = knn.predict(testcopy)
