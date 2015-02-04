# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:58:19 2015

@author: charlesmartens
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression   
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve


#CHALLENGE 1
house_votes = pd.read_csv('house-votes-84.data', header=None)
df1 = house_votes

df1.columns=['party', '1', '2', '3', '4', '5', '6', '7', '8', '9' , '10', '11', '12', '13', '14', '15', '16']

house_votes.head()
len(house_votes)

df1.replace('y', 1, inplace=True)
df1.replace('n', 0, inplace=True)


df1.replace('?', np.nan, inplace=True)

df1.dtypes

dftest = pd.DataFrame({'a': [4, 3, 2], 'b': [8, 4, 0]})
dftest['a'].mean()
dftest.replace(3, np.nan, inplace=True)
dftest.fillna(dftest.mean())

df1['2'].mean()
df1 = df1.fillna(df1.mean())


#CHALLENGE 2
train, test = train_test_split(df1, test_size = 0.25)
#dfTrain = pd.DataFrame(train)
#dfTest = pd.DataFrame(train)

yTrain = [l[0] for l in train]
xTrain = [l[1:] for l in train]
yTest = [l[0] for l in test]
xTest = [l[1:] for l in test]
#Y=[x[0] for x in train]
#X=[x[1:] for x in train]


#CHALLENGE 3
knn = KNeighborsClassifier(n_neighbors=5)  #init the model with knn=5
knn.fit(xTrain, yTrain)  #fits model on training set
yPredictions = knn.predict(xTest)

#accuracy_score(y_true, y_pred)
#accuracy_score(y_true, y_pred, normalize=False)
#accuracy_score(yTest, yPredictions, normalize=False)
accuracy_score(yTest, yPredictions)

#k=4 gives best predicion rate
for N in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=N)
    knn.fit(xTrain, yTrain)  #check: doesn't really fit anything, just puts these
    #xtrain and ytrain data into same system so that in next step (predict()) we 
    #can enter the test data and derive classification predictions.
    yPredictions = knn.predict(xTest)
    print accuracy_score(yTest, yPredictions)
    

#CHALLENGE 4
sk_logistic_model = LogisticRegression()
sk_logistic_model.fit(xTrain, yTrain)
yPredictionsLogistic = sk_logistic_model.predict(xTest)
print accuracy_score(yTest, yPredictionsLogistic)  #=.95. higher than knn (.93)
accuracyLogistic = accuracy_score(yTest, yPredictionsLogistic)  


#CHALLENGE 5

df1['party'].value_counts().plot(kind='bar')
#df1['party'].plot(kind='hist')


def outputDemocrat(xValues):
    newXvalues = ['democrat' for x in xValues]
    return newXvalues
 
yPredictedDemocrat = outputDemocrat(xTest)   
accuracyDemocratFunction = accuracy_score(yTest, yPredictedDemocrat)


def outputRepublican(xValues):
    newXvalues = ['republican' for x in xValues]
    return newXvalues
 
yPredictedRepublican = outputRepublican(xTest)   
accuracyRepublicanFunction = accuracy_score(yTest, yPredictedRepublican)


#CHALLENGE 6


def getKbyAccuracyKnn():
    ks = []
    accuracies = []
    for N in range(1,101):
        knn=KNeighborsClassifier(n_neighbors=N)
        knn.fit(xTrain, yTrain)  
        yPredictions = knn.predict(xTest)
        ks.append(N)        
        accuracies.append(accuracy_score(yTest, yPredictions))
    return ks, accuracies

ks, accuracyKnn = getKbyAccuracyKnn() 

plt.plot(ks, accuracyKnn, label='knn accuracy')
plt.plot(ks, [accuracyDemocratFunction for i in ks], label='democrat function accuracy')
plt.plot(ks, [accuracyRepublicanFunction for i in ks], label='republican function accuracy')
plt.plot(ks, [accuracyLogistic for i in ks], label='logistic accuracy')
plt.legend(loc='center left', bbox_to_anchor=(1, .5))


#CHALLENGE 7

#can enter entire data set and it splits it up into diff test and training sets
#so need to take df and turn into values/arrays?
valuesAll = df1.values
yAll = [l[0] for l in valuesAll]
xAll = [l[1:] for l in valuesAll]

#PART 1 - LOGISTIC
train_sizes, train_accuracy, test_accuracy = learning_curve(LogisticRegression(), xAll, yAll, train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], cv=10)
#cv signals how many folds in the cross validation
#are these returning errors or how well the test y is being predicted??

train_cv_accuracy_means = np.mean(train_accuracy, axis=1)  #axis=1 means getting mean along the columns. (assume axis=1 in pandas is same?)
test_cv_accuracy_means = np.mean(test_accuracy, axis=1)

plt.plot(train_sizes, train_cv_accuracy_means, label='training accuracy')
plt.plot(train_sizes, test_cv_accuracy_means, label='testing accuracy')
plt.legend()

#not getting very poor test accuracy with low m, like expected. but probably
#because the decision boundary is pretty clear, prediction accuracy is
#really high. for a more complex model with lower avg accuracy, might
#expect that the test accuracy would be very poor for small m -- i.e., for
#predicting accuracy on the test set from just a few points on the traning set.

#print 'mean sq error: ', math.sqrt(metrics.mean_squared_error(Y_test,Y_pred_test))

#PART 2 - KNN
train_sizes, train_accuracy_knn, test_accuracy_knn = learning_curve(KNeighborsClassifier(n_neighbors=5), xAll, yAll, train_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], cv=10)
train_cv_accuracy_means_knn = np.mean(train_accuracy_knn, axis=1)  #axis=1 means getting mean along the columns. (assume axis=1 in pandas is same?)
test_cv_accuracy_means_knn = np.mean(test_accuracy_knn, axis=1)

plt.plot(train_sizes, train_cv_accuracy_means_knn, label='training accuracy')
plt.plot(train_sizes, test_cv_accuracy_means_knn, label='testing accuracy')
plt.legend()

#the training accuracy here is essentially taking each training point and 
#treating it like a test point and using the rest of the accuracy data to 
#classify it. but conceptually regression is like this too in that it takes
#x-values from training data and uses to fit the line and then measures 
#with training ys.


#CHALLENGE 8

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()  #init the model with knn=5
gnb.fit(xTrain, yTrain)  #fits model on training set
yPredictions_gnb = gnb.predict(xTest)
accuracy_score(yTest, yPredictions_gnb)

from sklearn.svm import SVC
svMachine = SVC()  #init the model with knn=5
svMachine.fit(xTrain, yTrain)  #fits model on training set
yPredictions_svm = svMachine.predict(xTest)
accuracy_score(yTest, yPredictions_svm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()  #init the model with knn=5
dtc.fit(xTrain, yTrain)  #fits model on training set
yPredictions_dtc = dtc.predict(xTest)
accuracy_score(yTest, yPredictions_dtc)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()  #init the model with knn=5
rfc.fit(xTrain, yTrain)  #fits model on training set
yPredictions_rfc = rfc.predict(xTest)
accuracy_score(yTest, yPredictions_rfc)


#CHALLENGE 9

from sklearn.cross_validation import cross_val_score
#call the cross_val_score helper function on the estimator and the dataset

scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), xAll, yAll, cv=5)
scores.mean()   #returns an accuracy score for each of the 5 cross-validation folds
                #so take the mean here to get best sense for accuracy 

#can also get the mean sq error here instead of accuracy, if using regression
#http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter



