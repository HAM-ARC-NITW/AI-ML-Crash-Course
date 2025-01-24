# Introduction to Logistic Regression

## 1. What is Logistic Regression?

Logistic regression is a **classification algorithm** that predicts the probability of a binary outcome (0 or 1). Despite its name, it is used for classification tasks, not regression. It estimates the probability that an instance belongs to a particular class using a **sigmoid function**.

---

## 2. Difference Between Linear and Logistic Regression

- **Linear Regression**: Used for predicting continuous values.
- **Logistic Regression**: Used for predicting probabilities and classifying into categories (usually binary).

---

## 3. Sigmoid Function

The sigmoid function transforms the linear equation into a probability value:

$$
\\sigma(z) = \\frac{1}{1 + e^{-z}}
$$

The output is a probability between 0 and 1. The decision boundary is typically set at 0.5: if the output is greater than 0.5, the instance is classified as 1 (positive), otherwise, it is 0 (negative).

---

## 4. Logistic Regression Hypothesis Function

The logistic regression hypothesis is:

$$
h_{\\theta}(x) = \\sigma(\\theta_0 + \\theta_1 x_1 + \\dots + \\theta_n x_n)
$$

Where `\\theta` represents the learned parameters, and `x` represents the input features.

---

## 5. Cost Function (Log Loss)

The cost function for logistic regression, called **log loss** or **binary cross-entropy**, is:

$$
J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_{\\theta}(x^{(i)})) + (1 - y^{(i)}) \\log(1 - h_{\\theta}(x^{(i)})) \\right]
$$

This function helps to minimize the difference between the predicted probabilities and the actual labels.

---

## 6. Gradient Descent and Optimization

Gradient descent is used to find the optimal parameters \( \\theta \) by iteratively adjusting them in the direction that reduces the cost function. In **scikit-learn**, optimization is handled internally using algorithms like **L-BFGS**.

---

## 7. Regularization in Logistic Regression

Regularization helps prevent **overfitting** by penalizing large weights:

- **L1 (Lasso)**: Adds an absolute value penalty on the weights.
- **L2 (Ridge)**: Adds a squared value penalty on the weights.

---

## 8. Multiclass Classification (Optional)

Logistic regression can be extended to handle multiclass problems using strategies like **one-vs-rest (OvR)** or **softmax regression** for multi-class classification.

---

# Practical Implementation of Logistic Regression with Scikit-learn

## Example Code:

```python
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
data = load_breast_cancer()
X = data['data']
y = data['target']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 7: Print a detailed classification report
print(classification_report(y_test, y_pred))

# Step 8: Predict probabilities (Optional)
probabilities = model.predict_proba(X_test)
print(probabilities[:5])
```
