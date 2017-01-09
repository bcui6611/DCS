from sklearn import svm, datasets, metrics, preprocessing, linear_model
from sklearn.externals import joblib
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

"""
Pre-processing + SVM
"""
print('Pre-processing + SVM Learning... Fitting... ')
# 1- Scaling:
max_abs_scaler = preprocessing.MaxAbsScaler()
trainData = max_abs_scaler.fit_transform(X=trainData)
testData = max_abs_scaler.transform(X=testData)

svm_clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                        max_iter=1000)
svm_obj = svm_clf.fit(X=trainData, y=trainLabels)

"""
print("Plotting.. Before")
xx = np.linspace(-1, 1)
w = svm_clf.coef_[0]
a = -w[0] / w[1]
yy = a * xx - (svm_clf.intercept_[0]) / w[1]

# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
plt.scatter(trainData[:, 0], trainData[:, 1], c='blue', cmap=plt.cm.Paired)
plt.legend()
plt.axis('tight')
plt.show()
"""

print('Pre-processing + SVM Predicting... ')
predicted = svm_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("Score Accuracy Mean: %.4f " % svm_clf.score(X=testData, y=testLabels))

print("Pre-processing_SVM Saving in ... /Output/PRE_SVM_model.pkl")
joblib.dump(svm_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/PRE_SVM_model.pkl')

"""

print("Plotting.. After")
xx = np.linspace(-1, 1)
w = svm_clf.coef_[0]
a = -w[0] / w[1]
yy = a * xx - (svm_clf.intercept_[0]) / w[1]

# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
plt.scatter(trainData[:, 0], trainData[:, 1], c='blue', cmap=plt.cm.Paired)
plt.legend()
plt.axis('tight')
plt.show()


print('Plotting - Visualizing..')
fig, ax = plt.subplots()
ax.scatter(testLabels, predicted)
ax.plot([testLabels.min(), testLabels.max()], [testLabels.min(), testLabels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
"""