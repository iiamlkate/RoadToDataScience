import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set(color_codes = True)
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# %matplotlib inline

train = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_train.csv')
test = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_test.csv')

meanAge = np.mean(train.Age)
train.Age = train.Age.fillna(meanAge)

meanAge = np.mean(test.Age)
test.Age = test.Age.fillna(meanAge)
meanFare = np.mean(test.Fare)
test.Fare = test.Fare.fillna(meanFare)

# print(train.info())
# print(test.info())

train = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked', 'Pclass'], axis = 1)
train = train.dropna(axis = 0)

test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked', 'Pclass'], axis = 1)
test = test.dropna(axis = 0)


# print(test.describe())
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
# train['Embarked'] = train['Embarked'].map({'Q':0,'S':1,'C':2})

# meanEmbarked = np.mean(train.Embarked)
# train.Embarked = train.Embarked.fillna(meanEmbarked)

test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
# test['Embarked'] = test['Embarked'].map({'Q':0,'S':1,'C':2})

train['Sex'] = pd.to_numeric(train['Sex'], errors='ignore')
# train['Embarked'] = pd.to_numeric(train['Embarked'], errors='ignore')

test['Sex'] = pd.to_numeric(test['Sex'], errors='ignore')
# test['Embarked'] = pd.to_numeric(test['Embarked'], errors='ignore')

# print(train.info())
# print(test.info())

x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)

x = test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
test = pd.DataFrame(x_scaled)


x_train = train.iloc[:, :-1].values
y_train = train.iloc[:,-1].values


x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

# print(train)

# train.to_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\iamlkate_titanic_cleaned_data.csv')

#LogisticRegression
print('----------*****naive_bayes*****----------')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# print(y_pred)
print('classification_report:')
print(classification_report(y_test, y_pred))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))

#Accuracy score
print ('accuracy is', accuracy_score(y_pred, y_test))
print('')

print('----------*****GaussianNB*****----------')
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
classifier = KNeighborsClassifier(n_neighbors=3)
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

# np.savetxt("foo.csv", y_pred, delimiter=",")