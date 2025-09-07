======================

STEP 1: INSTALL DEPENDENCIES
======================

Goal: Install required Python libraries for data handling, neural network, plotting, and tuning.

Command:
pip install pandas numpy scikit-learn matplotlib tensorflow keras-tuner

Why: These libraries handle data processing, model building, hyperparameter tuning, and visualization.

------------------------------------------------------

======================

STEP 2: IMPORT LIBRARIES
======================

Goal: Load all Python functions and classes needed for the pipeline.

Example:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Why: Each library has a role:
- pandas / numpy → Data handling
- sklearn → Train/test split, scaling, encoding
- tensorflow.keras → Neural network layers and models
- keras_tuner → Hyperparameter tuning
- matplotlib → Plotting results

------------------------------------------------------

======================

STEP 3: LOAD & INSPECT DATA
======================

Goal: Verify the CSV file is readable and columns match expectations.

Example:
csv_path = r"C:\Users\UsersName\Downloads\ML_Data.csv"
df = pd.read_csv(csv_path)
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.isna().sum())

Why: Ensures your dataset is ready for preprocessing and checks for missing values.

------------------------------------------------------

======================

STEP 4: PREPROCESS DATA
======================

Goal: Transform raw CSV into features and labels ready for neural network training.

Steps:
- Feature Engineering:
    * Combine GAD1..GAD7 → ANXIETY
    * Combine DAST1..DAST10 → DRUG_MISUSE
- Label Mapping:
    * Map DRUG_MISUSE sum → SERVICE categories (No Intervention, BI, RT)
- Encoding:
    * Convert string labels → integers with LabelEncoder
- Train/Test Split:
    * Separate data: 60% train / 40% test
- Feature Scaling:
    * Standardize numeric features using StandardScaler

Small Code Snippet:
X_train, X_test, y_train, y_test, le = load_and_preprocess(csv_path)

Why: Neural networks require numeric, scaled features and integer labels for multiclass classification.

------------------------------------------------------

======================

STEP 5: BUILD NEURAL NETWORK
======================

Goal: Define the network architecture and training configuration.

Structure:
- Input layer → 2 hidden layers → output layer with 3 units (softmax)
- Hidden layers: relu activation
- Output layer: softmax activation
- Loss: sparse_categorical_crossentropy
- Optimizer: Adam
- Metric: accuracy

Small Code Snippet:
model = models.Sequential([
    layers.Dense(units=16, activation="relu", input_shape=(2,)),
    layers.Dense(units=8, activation="relu"),
    layers.Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

Why: Configures the neural network for multiclass classification and efficient learning.

------------------------------------------------------

======================

STEP 6: HYPERPARAMETER SEARCH & TRAINING
======================

Goal: Automatically try different architectures and learning rates to find the best model.

Small Code Snippet:
best_model, history = train_model(X_train, y_train)

Why: Saves time and optimizes network performance without manually guessing hidden layer sizes or learning rate.

------------------------------------------------------

======================

STEP 7: EVALUATE MODEL
======================

Goal: Measure performance and visualize results.

Steps:
- Test Accuracy → Evaluate performance on hold-out test set
- Confusion Matrix → Check where model misclassifies each class
- Training Curves → Visualize training vs validation accuracy and loss

Small Code Snippet:
evaluate_model(best_model, history, X_test, y_test, le)

Why: Confirms model quality and provides visual insights.

------------------------------------------------------

======================

STEP 8: RUN FULL PIPELINE
======================

Goal: Combine all steps in a main block for reproducible execution.

Small Code Snippet:
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_and_preprocess(csv_path)
    best_model, history = train_model(X_train, y_train)
    evaluate_model(best_model, history, X_test, y_test, le)

Why: Allows running the complete workflow with one command.

------------------------------------------------------
