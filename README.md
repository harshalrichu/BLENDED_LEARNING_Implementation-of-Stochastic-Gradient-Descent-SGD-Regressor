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


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

# Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

# Standardizing the data
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

# Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train,y_train)

# Making predictions
y_pred=sgd_model.predict(x_test)

# Evaluating model performance
mse=mean_squared_error(y_test,y_pred)

print("="*50)
print('Name: HARSHAL RICHU S')
print('Reg No:212225240049')
print(f"MSE: {mse:.4f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)

# Print model coefficients
print("Model Coefficients:")
print("Coefficiens:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()




```

## Output:
<img width="1907" height="780" alt="Screenshot 2026-03-28 184606" src="https://github.com/user-attachments/assets/cac98e8f-7efc-4b64-9d97-8246a3d99480" />
<img width="1914" height="759" alt="Screenshot 2026-03-28 184704" src="https://github.com/user-attachments/assets/4b02071d-e781-43e2-93ba-d829bbc33f39" />
<img width="1869" height="364" alt="Screenshot 2026-03-28 184735" src="https://github.com/user-attachments/assets/a93a4691-7a8c-48d6-a761-c5beb29da98c" />
<img width="1920" height="695" alt="Screenshot 2026-03-28 184810" src="https://github.com/user-attachments/assets/1c946431-74cd-4b98-9fc7-ba9dfab7b2ab" />
















## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
