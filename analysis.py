import numpy as np
import csv as csv
from datetime import datetime
import pylab as pl
#from scipy.stats import spearmanr
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import auc_score
#from sklearn.preprocessing import normalize
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2, RFECV
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import zero_one
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

start = datetime.now()

csv_file_object=csv.reader(open('test.csv'))
#header=csv_file_object.next()
test_records=[]
for row in csv_file_object:test_records.append(row)
test_records=np.array(test_records)
test_records=test_records.astype(np.float)

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

"""
import matplotlib.pyplot as plt
plt.boxplot(records)
plt.show()

from pylab import pcolor, show, colorbar, xticks, yticks
pcolor(np.corrcoef(records))
colorbar()
show()
"""

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
  records, cl, test_size=0.25, random_state=0)


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.01], 'C': [10, 4]},]
tuned_parameters2 = [{'kernel': ['poly'], 'gamma': [.001, .0001, .1, .01], 'C': [10, 5]},]
scores = [
    ('precision', precision_score),
    ('recall', recall_score),
]

for score_name, score_func in scores:
    print "# Tuning hyper-parameters for %s" % score_name
    print

    clf = GridSearchCV(SVC(), tuned_parameters, score_func=score_func)
    clf.fit(x_train, y_train, cv=5)

    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
       print "%0.3f (+/-%0.03f) for %r" % (
           mean_score, scores.std() / 2, params)
    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred_clf = y_test, clf.predict(x_test)
    print classification_report(y_true, y_pred_clf)
    print

    """
    print "----------------------------------------------------------"
    print "Second SVM"
    rclf = GridSearchCV(SVC(), tuned_parameters2, score_func=score_func)
    rclf.fit(x_train, y_train, cv=5)
    print "Best parameters set found on development set:"
    print
    print rclf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in rclf.grid_scores_:
       print "%0.3f (+/-%0.03f) for %r" % (
           mean_score, scores.std() / 2, params)
    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred_rclf = y_test, rclf.predict(x_test)
    print classification_report(y_true, y_pred_rclf)
    print
    """
"""
print "============================================================"

print "Final Report"
for weight in [.1, .25, .5, .75, .9]:
    y_pred = []
    for each in range(len(y_pred_clf)):
      y_pred.append(int(round(weight * y_pred_clf[each] + (1-weight) * y_pred_rclf[each])))
    print classification_report(y_true, y_pred)
"""
"""
svc_new = SVC(probability=True, C=10, kernel='rbf', gamma=.01)
svc_new.fit(x_train, y_train)
y_pred = svc_new.predict(test_records)

print 'Outputting'
open_file_object = csv.writer(open(
                              "simple" + str(datetime.now().isoformat()) +
                              ".csv", "wb"))
for item in y_pred:
    open_file_object.writerow([item])


print 'Done'
print datetime.now() - start
"""