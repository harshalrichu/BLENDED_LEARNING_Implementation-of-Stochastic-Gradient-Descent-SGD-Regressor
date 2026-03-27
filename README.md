# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and perform initial inspection; remove irrelevant columns and convert categorical variables into numerical form using one-hot encoding.
2. Separate the dataset into independent features (X) and target variable (y), and apply standard scaling to normalize the data.
3. Split the scaled data into training and testing sets using an 80:20 ratio.
4. Initialize the Stochastic Gradient Descent (SGD) Regressor and train the model using the training data.
5. Predict car prices on the test data, evaluate the model using MSE, MAE, and R² score, and visualize actual versus predicted values.
## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#Load The Dataset

data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

# Data PreProcessing
# Dropping Unnecessary and handling categorical Variables
data = data.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data,drop_first=True)

#Spliting the data into features and target variable

X=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
#scaler=StandardScaler()
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\n=== Test Set Performance===")
print(f"MSE:",mse)
print(f"MAE:",mae)
print(f"R2:",r2)
r2=r2_score(y_test,y_pred)
print("Name:Harshal Richu S")
print("Reg No:25008177")




```

## Output:
<img width="1920" height="1020" alt="Screenshot 2026-02-24 205108" src="https://github.com/user-attachments/assets/7a5afff9-445d-4b6b-be95-a8ab14f98603" />
<img width="1920" height="1020" alt="Screenshot 2026-02-24 205117" src="https://github.com/user-attachments/assets/ffc40a10-378d-40dc-82e4-780a80e68f1d" />
<img width="1920" height="1020" alt="Screenshot 2026-02-24 205125" src="https://github.com/user-attachments/assets/a3aea035-5d80-42f3-b4d7-595947b07b69" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/747fe455-8149-42a8-a0f2-de355d681243" />









## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
