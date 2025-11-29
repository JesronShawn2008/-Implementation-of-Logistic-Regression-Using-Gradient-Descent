# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, convert the placement status to binary values, and extract the selected feature along with the target variable.

2. Normalize the chosen feature and initialize the logistic regression parameters (slope, intercept, learning rate, iterations).

3. Apply gradient descent by computing predictions using the sigmoid function, calculating gradients, and updating parameters repeatedly.

4. Output the final learned values and use them to predict placement probability for a given test input.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
import pandas as pd
import numpy as np

df = pd.read_csv("Placement_Data.csv")

df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

X = df["etest_p"].values
y = df["status"].values

X = (X - X.mean()) / X.std()

m = 0.0
c = 0.0
alpha = 0.01
iterations = 2000

n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(iterations):
    z = m * X + c
    h = sigmoid(z)

    dm = (1/n) * np.sum((h - y) * X)
    dc = (1/n) * np.sum(h - y)

    m = m - alpha * dm
    c = c - alpha * dc

    if i < 20:   # print first few iterations
        loss = -(1/n) * np.sum(y*np.log(h+1e-9) + (1-y)*np.log(1-h+1e-9))
        print(f"Iteration {i}, Loss={loss:.4f}, m={m:.4f}, c={c:.4f}")

print("\nFinal Slope m:", m)
print("Final Intercept c:", c)

def predict(e):
    e = (e - df["etest_p"].mean()) / df["etest_p"].std()
    prob = sigmoid(m*e + c)
    return 1 if prob >= 0.5 else 0, prob

test_value = 75
label, prob = predict(test_value)

print(f"\nPrediction for etest_p = {test_value}: {label}  (Probability = {prob:.4f})")

Developed by: Jesron Shawn C J
RegisterNumber:  25012933
*/
```

## Output:
<img width="1027" height="543" alt="image" src="https://github.com/user-attachments/assets/127bc2a6-3979-4d1a-8858-a05ce7b1bba4" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

