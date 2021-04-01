#Machine learning model for classifying survival of passengers in
#given tittanic data set(train.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

ship_data = pd.read_csv("train.csv")
#removing all the null value
ship_data.dropna(inplace=True)
ship_data.drop("Cabin", axis=1, inplace=True)

#removing possible string values to binary sex,embark

sex = pd.get_dummies(ship_data["Sex"], drop_first=True)
pclass  =pd.get_dummies(ship_data["Pclass"], drop_first=True)
embrk = pd.get_dummies(ship_data["Embarked"], drop_first=True)

#adding the all the above values into the dataset

ship_data.drop(["Sex", "Pclass", "Embarked", "PassengerId", "Name","Ticket"], axis=1,inplace=True)
ship_data  =pd.concat([ship_data,sex,pclass,embrk],axis=1)

#Training the dataset
x = ship_data.drop("Survived", axis=1)
y = ship_data["Survived"]

#spliting dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.45, random_state = 1)

#building our model
model = LogisticRegression()
model.fit(x_train, y_train)

suvive_predict = model.predict(x_test)

#printing accuracy score

print(accuracy_score(y_test,suvive_predict))