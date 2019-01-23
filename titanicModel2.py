import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set(color_codes = True)
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
# %matplotlib inline

train = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_train.csv')
test = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_test.csv')


train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknow', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)

train.Embarked = train.Embarked.fillna("S")

train = train.drop(['Cabin', 'Ticket'], axis = 1)



combine = [train,test]
for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False)

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
	dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
	dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)
print(train.head())
# print(pd.crosstab(train['Title'], train['Sex']))

mr_age = train[train['Title']==1]['AgeGroup'].mode()
miss_age = train[train['Title']==2]['AgeGroup'].mode()
mrs_age = train[train['Title']==3]['AgeGroup'].mode()
master_age = train[train['Title']==4]['AgeGroup'].mode()
royal_age = train[train['Title']==5]['AgeGroup'].mode()
rare_age = train[train['Title']==6]['AgeGroup'].mode()

age_title_mapping = {1:'Young Adult', 2:'Student', 3:'Adult', 4:'Baby', 5:'Adult', 6:'Adult'}

for x in range(len(train['AgeGroup'])):
	if train['AgeGroup'][x] == 'Unknow':
		train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
	if test['AgeGroup'][x] == 'Unknow':
		test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]

age_mapping = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()
train = train.drop(['Age', 'Name'], axis = 1)
test = test.drop(['Age', 'Name'], axis = 1)

sex_mapping = {'male':0, 'female':1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train['Embarked'] = train['Embarked'].map({'Q':0,'S':1,'C':2})
test['Embarked'] = test['Embarked'].map({'Q':0,'S':1,'C':2})

for x in range(len(test['Fare'])):
	if pd.isnull(test['Fare'][x]):
		pclass = test['Pclass'][x]
		test['Fare'][x] = round(train[train['Pclass'] == pclass]['Fare'].mean(), 4)

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


predictors = train.drop(['Survived', 'PassengerId'], axis = 1)
target = train['Survived']
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

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