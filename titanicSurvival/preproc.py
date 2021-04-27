import pathlib as pt
import pandas as pd
import numpy as np
pd.options.display.max_columns = 15
# read data
train_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv('datasets/test.csv')
# drop the columns that have low predictive value (in my opinion)
train_data.drop(['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
# print(train_data)
# get the indexes of NaN values
age_nan_index = train_data.index[train_data['Age'].isnull() == True].tolist()
embarked_nan_index = train_data.index[train_data['Embarked'].isnull() == True].tolist()
age_nan_index2 = test_data.index[test_data['Age'].isnull() == True].tolist()
embarked_nan_index2 = test_data.index[test_data['Embarked'].isnull() == True].tolist()
fare_nan_index = test_data.index[test_data['Fare'].isnull() == True].tolist()
# apply mean imputation on fare for only test data
fare_mean = test_data['Fare'].mean()
for i in fare_nan_index:
    test_data.at[i, 'Fare'] = fare_mean
# apply mean imputation on age
age_mean = train_data['Age'].mean()
age_mean2 = test_data['Age'].mean()
for i in age_nan_index2:
    test_data.at[i, 'Age'] = age_mean2
for i in age_nan_index:
    train_data.at[i, 'Age'] = age_mean
# print(test_data['Age'].isnull().any())
# apply mode imputation on embarked
embarked_mode = train_data['Embarked'].mode()
embarked_mode2 = test_data['Embarked'].mode()
for i in embarked_nan_index2:
    test_data.at[i, 'Embarked'] = embarked_mode2
for i in embarked_nan_index:
    train_data.at[i, 'Embarked'] = embarked_mode
# print(train_data.isnull().any())  # now we know there is no NaN value left
train_data['Sex'] = train_data['Sex'].astype('category')
# print(train_data['Embarked'])
train_data['Embarked'] = train_data['Embarked'].astype('category')
print(train_data['Embarked'])
'''
train_data.to_csv('train_ready.csv', index=False)
test_data.to_csv('test_ready.csv', index=False)
'''