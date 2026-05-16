# Dropout Regularization using TensorFlow and Keras

## 📌 Introduction
This project demonstrates the implementation of **Dropout Regularization** in an Artificial Neural Network (ANN) using TensorFlow and Keras. Dropout is a regularization technique used to reduce overfitting and improve the generalization capability of deep learning models.

---

## 🧠 What is Dropout Regularization?
Dropout is a technique where randomly selected neurons are ignored during training. This means that their contribution is temporarily removed during forward and backward propagation.

The main purpose of dropout is to:
- Reduce overfitting
- Improve model generalization
- Prevent dependency on specific neurons

---

## 🚀 Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

---

## 📂 Project Workflow
1. Import Libraries
2. Load Dataset
3. Data Preprocessing
4. Feature Scaling
5. Train-Test Split
6. Build ANN Model
7. Apply Dropout Layers
8. Compile Model
9. Train Model
10. Evaluate Model

---

## 📦 Import Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
📊 Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
🏗️ Build ANN Model with Dropout
model = keras.Sequential([

    keras.layers.Dense(60, input_dim=8, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(15, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])
⚙️ Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
 Train the Model
model.fit(X_train, y_train, epochs=100)
📈 Evaluate the Model
model.evaluate(X_test, y_test)
 Advantages of Dropout
Reduces overfitting
Improves generalization
Prevents neuron dependency
Makes the model more robust
 Limitations
Training may become slower
High dropout values can reduce performance
Requires proper tuning
 Conclusion

Dropout Regularization is one of the most effective techniques used in deep learning to prevent overfitting. By randomly disabling neurons during training, the model learns more generalized patterns and performs better on unseen data.
