import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Trained Model and Test Data ---
# ------------------------------------------------

# Load the model and encoder you saved during training
model = joblib.load("human_activity_model.pkl")
le = joblib.load("label_encoder.pkl")
print("--- Model and Label Encoder Loaded ---")

# IMPORTANT: Replace this with the path to your test file
TEST_DATA_PATH = "test1.csv" 

# Define the column names, same as before
columns = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'label']

# Load the new test data, SKIPPING the first row (the header)
test_df = pd.read_csv(TEST_DATA_PATH, header=None, names=columns, skiprows=1)

print(f"\n--- Loaded test.csv with {len(test_df)} rows ---")


# --- 2. Preprocess the Test Data ---
# -------------------------------------
# Apply the EXACT SAME steps as in the training script.

# a) Noise Reduction (Signal Smoothing)
sensor_cols = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
window_size = 5

for col in sensor_cols:
    test_df[col + '_filtered'] = test_df[col].rolling(window=window_size, min_periods=1).mean()

print("\n--- Step 2a: Noise Reduction Applied ---")


# b) Feature Extraction
SAMPLING_RATE = 2
TIME_STEPS = 4
STEP_SIZE = 2

features = []
labels = []
filtered_cols = [col + '_filtered' for col in sensor_cols]

for i in range(0, len(test_df) - TIME_STEPS, STEP_SIZE):
    window_data = test_df[filtered_cols].iloc[i : i + TIME_STEPS]
    window_labels = test_df['label'].iloc[i : i + TIME_STEPS]
    
    current_label = window_labels.mode()[0]
    
    feature_row = []
    for sensor in filtered_cols:
        feature_row.append(window_data[sensor].mean())
        feature_row.append(window_data[sensor].std())
        feature_row.append(window_data[sensor].min())
        feature_row.append(window_data[sensor].max())
        feature_row.append(np.sqrt(np.mean(window_data[sensor]**2)))
    
    features.append(feature_row)
    labels.append(current_label)

# Create the final feature DataFrame for the test set
feature_columns = []
for sensor in sensor_cols:
    for stat in ['mean', 'std', 'min', 'max', 'rms']:
        feature_columns.append(f"{sensor}_{stat}")

df_test_features = pd.DataFrame(features, columns=feature_columns)
df_test_features['label'] = labels

print(f"--- Step 2b: Feature Extraction Complete. Created {len(df_test_features)} feature windows. ---")


# --- 3. Make Predictions and Evaluate ---
# ----------------------------------------

# Separate features (X) and labels (y)
X_unseen = df_test_features.drop('label', axis=1)
y_unseen_labels = df_test_features['label']

# Encode the string labels from the test set into numbers
y_unseen_encoded = le.transform(y_unseen_labels)

# Use the loaded model to make predictions
y_pred_encoded = model.predict(X_unseen)

# Get the unique label indexes that are actually in this test set
unique_label_indices = np.unique(y_unseen_encoded)

# Get the corresponding string names for only those unique labels
target_names_for_report = le.inverse_transform(unique_label_indices)

# Evaluate the model's performance on the new data
accuracy = accuracy_score(y_unseen_encoded, y_pred_encoded)
print(f"\n--- Model Evaluation on test.csv ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
# Use the dynamic labels and names we just created
print(classification_report(
    y_unseen_encoded, 
    y_pred_encoded, 
    labels=unique_label_indices, 
    target_names=target_names_for_report
))

# Visualize the Confusion Matrix
# We also use the dynamic names for the plot labels
cm = confusion_matrix(y_unseen_encoded, y_pred_encoded, labels=unique_label_indices)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names_for_report, 
            yticklabels=target_names_for_report)
plt.title('Confusion Matrix on Unseen Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()