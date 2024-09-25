
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.                                                                                                                                  Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sujithra D
RegisterNumber:  212222220052

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
```
## dataset:
```
dataset
```
## output:
![image](https://github.com/user-attachments/assets/f2507ddc-5ba5-429c-b60c-d48d11c1b392)
## head and tail:
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![image](https://github.com/user-attachments/assets/97e2a90d-20b6-4838-b9ee-0a9630840a0e)
## x and y value
```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```
## output
![image](https://github.com/user-attachments/assets/fe674dc2-3cbe-4a03-9fdb-db4ece326185)
## Predication values of X and Y
## program
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
x_test.shape
```
## output:
![image](https://github.com/user-attachments/assets/6a2bd9a6-e41d-4e5b-9b32-5b46e87d8133)
## program
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
## output
![image](https://github.com/user-attachments/assets/c670ddcb-c30a-4e58-9fb4-b558986abaa6)
## program
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```
## output:
![image](https://github.com/user-attachments/assets/8116ebb1-7ce9-4d4c-94c3-0109dd3c16ba)
## MSE,MAE and RMSE
```
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## output
![image](https://github.com/user-attachments/assets/b6090e04-34a6-4246-9474-4306f37510e9)
## Training Set
```
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='purple')
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,reg.predict(x_test),color='yellow')
plt.title('test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
## output
![Screenshot 2024-08-28 103715](https://github.com/user-attachments/assets/26ff5be6-f39c-430e-a340-c8735fd6a0e2)
![image](https://github.com/user-attachments/assets/339df37c-005b-4d02-b6e6-da8c2b77e167)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
