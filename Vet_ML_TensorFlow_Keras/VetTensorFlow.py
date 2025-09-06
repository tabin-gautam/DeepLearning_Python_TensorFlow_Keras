# data handling
import pandas as pd
import numpy as np
# splitting,encoding ,scalling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# plotting
import matplotlib.pyplot as plt
# model building and training
import tensorflow as tf
from tensorflow.keras import layers, models
# hyperparameter search
import keras_tuner as kt
# evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data Loading
csv_path = r"C:\Users\Tabin\Downloads\VetSampleData.csv"
df = pd.read_csv(csv_path)


# Data Processing/Cleaning
def load_and_preprocess(path):
    # read csv file and remove the Rows with NA
    data = pd.read_csv(path).dropna()
    # Feature engineering:
    data["ANXIETY"] = data[["GAD1", "GAD2", "GAD3", "GAD4", "GAD5", "GAD6", "GAD7"]].sum(axis=1)
    data["DRUG_MISUSE"] = data[["DAST1", "DAST2", "DAST3", "DAST4", "DAST5",
                                "DAST6", "DAST7", "DAST8", "DAST9", "DAST10"]].sum(axis=1)

    # Recode DrugMisUse into service
    def recode_service(x):
        if x == 0:
            return "No Intervention"
        elif x in [1, 2]:
            return "BI"
        else:
            return "RT"

    data["SERVICE"] = data["DRUG_MISUSE"].apply(recode_service)
    data = data[["AGE", "ANXIETY", "SERVICE"]]

    # Convert SERVICE to numeric labels (0,1,2) using LabelEncoder
    le = LabelEncoder()
    data["SERVICE"] = le.fit_transform(data["SERVICE"])

    # Split into features (x) and target (y), then train/test split
    x = data[["AGE", "ANXIETY"]].values
    y = data["SERVICE"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)

    # Standardize numeric features (mean=0, std=1) — For NN training
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, le


# Model Build
def build_model(hp):
    model = models.Sequential([
        # first dense layer; units are tunable (hp.Int)
        layers.Dense(units=hp.Int("units1", 8, 64, step=8), activation="relu", input_shape=(2,)),
        # second dense layer; tunable units
        layers.Dense(units=hp.Int("units2", 4, 32, step=4), activation="relu"),
        # final layer -> 3 outputs for 3 classes, softmax for probabilities
        layers.Dense(3, activation="softmax")
    ])
    # compile: Adam optimizer with tunable learning rate; sparse_categorical_crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [0.001, 0.01, 0.1])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# model train and hyperparameter search
def train_model(x_train, y_train):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=5,  # number of different hyperparam configs to try
        executions_per_trial=1,  # how many times to train each config
        overwrite=True,
        directory="tuner_results",
        project_name="Vet_ML_TensorFlow_Keras"
    )
    # Run the hyperparameter search (this trains models). validation_split is used for tuner.
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2, verbose=1)
    best_model = tuner.get_best_models(num_models=1)[0]
    # re-train (or continue training) the best model on the training set to collect history
    history = best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return best_model, history


# Evaluation and Plot
def evaluate_model(model, history, x_test, y_test, le):
    # Final test evaluation
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    # Predictions and confusion matrix
    y_prob = model.predict(x_test)  # shape (n_samples, 3)
    y_pred = y_prob.argmax(axis=1)  # class indices
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.show()

    # Plot training curves (accuracy & loss)
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.show()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, le = load_and_preprocess(csv_path)
    best_model, history = train_model(x_train, y_train)
    evaluate_model(best_model, history, x_test, y_test, le)


