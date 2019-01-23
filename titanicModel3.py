import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

train_df = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_train.csv')
test_df = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_test.csv')


print(train_df.isnull().sum())

train_df = train_df.drop(['Cabin'], axis = 1)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

min_age = train_df['Age'].min()
max_age = train_df['Age'].max()
null_values_count = train_df['Age'].isnull().sum()
age_fill_values = np.random.randint(min_age, max_age, size = null_values_count)
train_df['Age'][np.isnan(train_df['Age'])] = age_fill_values

train_df['Embarked'] = train_df['Embarked'].fillna('S')

#---------------- Pclass and survival ----------------
train_df['High Class'] = np.where(train_df['Pclass'] == 1, 1, 0)
train_df['Median Class'] = np.where(train_df['Pclass'] == 2, 1, 0)

#---------------- Sex and survival ----------------
train_df = train_df.replace({'Sex': {'male':0, 'female':1}})

#---------------- Age and survival ----------------
age = train_df[['Age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
train_df['Age'] = pd.DataFrame(age_scaled)
print(train_df.head())

#---------------- SibSp and survival ----------------
train_df['SibSp'] = np.where(train_df['SibSp'] > 0, 1, 0)

#---------------- Parch and survival ----------------
train_df['Parch'] = np.where(train_df['Parch'] > 0, 1, 0)

#---------------- Parch and survival ----------------
fare = train_df[['Fare']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(fare)
train_df['Fare'] = pd.DataFrame(fare_scaled)

#---------------- Embarked and survival ----------------
train_df['Embarked C'] = np.where(train_df['Embarked'] == 'C', 1, 0)
train_df['Embarked Q'] = np.where(train_df['Embarked'] == 'Q', 1, 0)
train_df['Embarked S'] = np.where(train_df['Embarked'] == 'S', 1, 0)

final_train_df = train_df[['Survived', 'High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked C', 'Embarked Q', 'Embarked S']]

independent_v_train = final_train_df[['High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked C', 'Embarked Q', 'Embarked S']]
dependent_v_train = final_train_df['Survived']

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(independent_v_train, dependent_v_train)
print('SCORE: ', rf.score(independent_v_train, dependent_v_train))

dt = DecisionTreeClassifier()
dt.fit(independent_v_train, dependent_v_train)
print('SCORE: ', dt.score(independent_v_train, dependent_v_train))

gb = GradientBoostingClassifier()
gb.fit(independent_v_train, dependent_v_train)
print('SCORE: ', gb.score(independent_v_train, dependent_v_train))

test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test_df['High Class'] = np.where(test_df['Pclass'] == 1, 1, 0)
test_df['Median Class'] = np.where(test_df['Pclass'] == 2, 1, 0)
test_df = test_df.replace({'Sex': {'male':0, 'female':1}})
min_age = test_df['Age'].min()
max_age = test_df['Age'].max()
null_values_count = test_df['Age'].isnull().sum()
age_fill_values = np.random.randint(min_age, max_age, size = null_values_count)
test_df['Age'][np.isnan(test_df['Age'])] = age_fill_values
age = test_df[['Age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
test_df['Age'] = pd.DataFrame(age_scaled)
test_df['SibSp'] = np.where(test_df['SibSp'] > 0, 1, 0)
test_df['Parch'] = np.where(test_df['Parch'] > 0, 1, 0)
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
fare = train_df[['Fare']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(fare)
test_df['Fare'] = pd.DataFrame(fare_scaled)
test_df['Embarked'] = test_df['Embarked'].fillna('S')
test_df['Embarked C'] = np.where(test_df['Embarked'] == 'C', 1, 0)
test_df['Embarked Q'] = np.where(test_df['Embarked'] == 'Q', 1, 0)
test_df['Embarked S'] = np.where(test_df['Embarked'] == 'S', 1, 0)
final_test_df = test_df[['High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked C', 'Embarked Q', 'Embarked S']]

independent_v_test = final_test_df
dependent_v_test_predict = gb.predict(independent_v_test)
survival_df = pd.DataFrame(dependent_v_test_predict)
test_get_id = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\set_test.csv')
prediction_df = pd.DataFrame(test_get_id['PassengerId'])
prediction_df['Survived'] = survival_df

prediction_df.to_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataTitanic\iamlkate_titanic_result_v3.csv', index=False)