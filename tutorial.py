import csv as csv
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
# Reading in training data for Kaggle sci kit competition
csv_file_object=csv.reader(open('train.csv'))
header=csv_file_object.next()
records=[]
for row in csv_file_object:records.append(row)
records=np.array(records)
records=records.astype(np.float)

csv_file_object=csv.reader(open('trainLabels.csv'))
header=csv_file_object.next()
cl=[]
for row in csv_file_object:cl.append(row)
cl=np.array(cl)
cl=cl.astype(np.int8)
cl=cl.reshape(999,)
tr_ex=np.size(cl)

#Need to use 70% of the data for training and 30% for testing
n_train=int(.7*tr_ex)
x_train,x_test=records[:n_train,:],records[n_train:,:]
y_train,y_test=cl[:n_train],cl[n_train:]

#SVM code

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# I tried different models, but this one with c=10 and gamma=.01 gives
# gives the SVM benchmark score.
clf=svm.SVC(C=10.0,gamma=.01,kernel='rbf',probability=True)
clf.fit(x_train,y_train)
print clf.n_support_
y_pred1=clf.predict(x_test)
gau_score=clf.score(x_test,y_test)
print"This is the score for rbf model",gau_score
cm1=confusion_matrix(y_test,y_pred1)
print "This is the confusion matrix for rbf model",(cm1)

print "finished"