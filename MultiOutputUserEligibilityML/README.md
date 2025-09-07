🏥 AI/ML Multi Output Neural Network & Anomaly Detection

This project demonstrates how to use TensorFlow to predict eligibility for public services (like Medicaid, Food Assistance, WIC) and to detect unusual or suspicious applications with anomaly detection.

🚀 Project Overview

Data Inputs: Age, Income, Citizenship, Household Size, Disability, Prior Service History

Outputs: Eligibility for Medicaid, Food, and WIC

Models Used:

Multi-Output Neural Network → predicts eligibility decisions.

Autoencoder → detects anomalies (possible fraud/unusual cases).

📂 Files

eligibility_ai_pipeline.py → Main script (training, prediction, anomaly detection).

mock_user_data.csv → Example dataset (mocked user info + eligibility labels).

README.md → This documentation.

⚙️ Setup Instructions
1. Install Dependencies
pip install pandas numpy matplotlib scikit-learn tensorflow keras-tuner

2. Open in PyCharm

Create a new Python Project in PyCharm.

Add eligibility_ai_pipeline.py and mock_user_data.csv to the project.

3. Update File Path

Inside eligibility_ai_pipeline.py, change this line if needed:

csv_path = r"C:\Users\UsersName\Downloads\mock_user_data.csv"


to match where your CSV is stored.

4. Run Script

Click Run ▶️ in PyCharm.
The script will:

Train the multi-output model.

Print accuracy results.

Show training curves.

Train anomaly detection autoencoder.

Print anomaly count and plot error distribution.

📊 How It Works
🔹 Multi-Output Neural Network

Input → Age, Income, Citizenship, Household size, Disability, etc.

Hidden layers → Learn complex patterns.

Output → 3 predictions (Medicaid, Food, WIC) using sigmoid activation.

Loss → binary_crossentropy since each program is independent.

✅ Example prediction:

{
  "eligible_medicaid": true,
  "eligible_food": true,
  "eligible_wic": false
}

🔹 Anomaly Detection (Autoencoder)

Learns normal user patterns by reconstructing input features.

If a new user’s reconstruction error is too high → flagged as anomaly.

Helps detect fraudulent or unusual applications.

✅ Example anomaly flag:

{
  "user_id": 123,
  "anomaly_flag": true
}