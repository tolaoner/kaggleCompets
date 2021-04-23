import pathlib as pt
import pandas as pd
import numpy as np
pd.options.display.max_columns=15
# read data
train_data = pd.read_csv("datasets/train.csv")
# drop the columns that have low predictive value (in my opinion)
train_data.drop(['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
# get the indexes of NaN values
age_nan_index = train_data.index[train_data['Age'].isnull() == True].tolist()
embarked_nan_index = train_data.index[train_data['Embarked'].isnull() == True].tolist()
# apply mean imputation on age
age_mean = train_data['Age'].mean()
for i in age_nan_index:
    train_data.at[i, 'Age']= age_mean
print(train_data['Age'].isnull().any())
# apply mode imputation on embarked
embarked_mode = train_data['Embarked'].mode()
for i in embarked_nan_index:
    train_data.at[i, 'Embarked']=embarked_mode
print(train_data['Embarked'].isnull().any())