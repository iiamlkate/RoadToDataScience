import sklearn
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set(color_codes = True)

data = pd.read_csv('D:\Stream\Stream_DC_SM10\Road to Data Scientist\dataLoan\loan.csv')
# print(data.head())

# print(data.shape)

print(data.info())

print(data.describe())
# print(data.isnull())