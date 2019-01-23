import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set(color_codes = True)
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
# %matplotlib inline

data = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\irisData\iris.csv')
# print(data.head())

# print(data.shape)

# print(data.info())

# print(data.describe())

# print(data.groupby('species').size())

# data.plot(kind='box', sharex=False, sharey=False)
# # plt.show()

# pd.scatter_matrix(data,figsize=(10,10))
# plt.show()

# sns.pairplot(data, hue='species')
# plt.show()

x = data.iloc[:, :-1].values
y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#LogisticRegression
print('----------*****naive_bayes*****----------')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

#Accuracy score
from sklearn.metrics import accuracy_score
print ('accuracy is', accuracy_score(y_pred, y_test))
print('')

print('----------*****naive_bayes*****----------')
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print ('accuracy is', accuracy_score(y_pred, y_test))
print('')

print('----------*****support vector machine\'s*****----------')
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print ('accuracy is', accuracy_score(y_pred, y_test))
print('')

print('----------*****K-Nearest Neighbours*****----------')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print ('accuracy is', accuracy_score(y_pred, y_test))
print('')

print('----------*****Decision Tree\'s*****----------')
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print ('accuracy is', accuracy_score(y_pred, y_test))