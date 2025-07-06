data cleaning

import pandas as pd
import numpy as np
#load dataset
dataset = pd.read_csv("student.csv")
#print(dataset.describe())
#random
dataset.sample(random_state = 3)
print("first five lines")
h1 = dataset.head()
print(h1)
h2 = dataset.tail()
#print(h2)
#print(h2)
#create independent and dependent variables
x = dataset.iloc[:,:-1].values
#print(x)
y = dataset.iloc[:,2].values
#print(y)
#handling missing data
misscount = dataset.isnull().sum()
#print(misscount)
data_1 = dataset.dropna(axis=0, how="all")
#print(data_1)
data_1=dataset.dropna(axis=0,how="any")
#print(data_1.isnull().sum())
data_1 = dataset.dropna(axis=0,how="all",subset=["CGPA"])
data_1.isnull().sum()
#print(data_1.isnull().sum())
data_1 = dataset.dropna(axis=0,how="all",subset=["Age"])
data_1.isnull().sum()
print(data_1.isnull().sum())
#calculate the average values
mean = dataset["CGPA"].mean()
mean=round(mean,2)#print values upto 2 decimal
dataset['CGPA']=dataset['CGPA'].fillna(mean)#fill null value mean
#print((dataset["CGPA"]))
#print((dataset.isnull().sum()))
mean1 = dataset["Age"].mean()
print(mean1)
mean1=round(mean1,2)
print(mean1)
dataset["Age"]=dataset["Age"].fillna(mean1)
print(dataset["Age"])
print((dataset.isnull().sum()))
