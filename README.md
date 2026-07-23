# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Imports
2. Load dataset
3. Prepare input X and target Y
4. Train/test split
5. Create and train the linear model
6. Predict on test set
7. Plots
8. Evaluation metrics
9. Predict new samples

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Charithrakshi K
RegisterNumber:  212224040053
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")



X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))


# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data

Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()


order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()



mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)



new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```




## Output:
<img width="332" height="362" alt="image" src="https://github.com/user-attachments/assets/71fc1f7d-9e42-4324-a266-7f7ba9e62f17" />
<img width="851" height="103" alt="image" src="https://github.com/user-attachments/assets/1aa940dc-84d9-4637-a8c1-eb1ff2346d84" />
<img width="446" height="56" alt="image" src="https://github.com/user-attachments/assets/3010252a-3ea7-4a17-bb2f-512426d86dba" />
<img width="317" height="92" alt="image" src="https://github.com/user-attachments/assets/ab0a72d1-e787-4f17-b9c9-cc3c5d52fa46" />
<img width="317" height="92" alt="image" src="https://github.com/user-attachments/assets/ab0a72d1-e787-4f17-b9c9-cc3c5d52fa46" />
<img width="666" height="478" alt="image" src="https://github.com/user-attachments/assets/4e4c1ec4-d510-4bda-a313-9226294a1b5b" />
<img width="665" height="487" alt="image" src="https://github.com/user-attachments/assets/78771390-f9af-4a16-a8e9-40b7867cd062" />
<img width="670" height="97" alt="image" src="https://github.com/user-attachments/assets/e0f7b0b4-78ef-45cd-8a15-e50a2e3074f1" />
<img width="527" height="48" alt="image" src="https://github.com/user-attachments/assets/f703e1da-8880-48d3-9103-0bb398b3d5d3" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
