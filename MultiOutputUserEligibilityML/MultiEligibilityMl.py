import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# STEP 1: Load & Preprocess
# ----------------------------
def load_and_preprocess(path):
    data = pd.read_csv(path).dropna()

    # Features and Labels
    x = data[["age", "income", "citizenship", "prior_services",
              "household_size", "has_disability",
              "previous_medicaid", "previous_wic"]].values

    y = data[["eligible_medicaid", "eligible_food", "eligible_wic"]].values

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, scaler


# ----------------------------
# STEP 2: Multi-Output Model
# ----------------------------
def build_multi_output_model(input_dim):
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="sigmoid")  # 3 outputs
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------
# STEP 3: Autoencoder for Anomaly Detection
# ----------------------------
def build_autoencoder(input_dim):
    autoencoder = models.Sequential([
        layers.Dense(16, activation="relu", input_shape=(input_dim,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


# ----------------------------
# STEP 4: Train + Evaluate
# ----------------------------
def run_pipeline(csv_path):
    # Load data
    x_train, x_test, y_train, y_test, scaler = load_and_preprocess(csv_path)
    input_dim = x_train.shape[1]

    # --- Multi-output classifier ---
    clf = build_multi_output_model(input_dim)
    history = clf.fit(
        x_train, y_train,
        epochs=30, batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate classification model
    loss, acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"\nMulti-Output Model Accuracy: {acc:.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend();
    plt.title("Classification Accuracy")
    plt.show()

    # --- Autoencoder ---
    autoencoder = build_autoencoder(input_dim)
    ae_history = autoencoder.fit(
        x_train, x_train,
        epochs=30, batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Reconstruction error
    reconstructions = autoencoder.predict(x_test)
    mse = np.mean(np.square(x_test - reconstructions), axis=1)

    # Threshold for anomaly detection
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    print(f"\nAnomaly threshold: {threshold:.4f}")
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(x_test)} samples")

    # Plot anomaly scores
    plt.hist(mse, bins=50)
    plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE");
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    csv_path = r"C:\Users\Tabin\Downloads\applicant_mock_data.csv"  # Update path
    run_pipeline(csv_path)
