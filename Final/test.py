import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load the model
model = load_model("lstm_model.h5")

# Load the test set
test_df = pd.read_csv("dataset/test_set.csv")

# Extract feature columns (x, y, z)
feature_cols = [col for col in test_df.columns if col.startswith(('x', 'y', 'z'))]

# Reshape function
def reshape_for_lstm(df):
    X = df[feature_cols].values
    X = X.reshape((X.shape[0], 5, 3))  # 5 timesteps, 3 features
    return X

# Prepare test data
X_test = reshape_for_lstm(test_df)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(test_df['activity_id'])  # Make sure same order as during training
y_test = label_encoder.transform(test_df['activity_id'])

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

# Decode back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# Metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# print("\nClassification Report:")
# print(classification_report(y_true_labels, y_pred_labels))

# cm = confusion_matrix(y_true_labels, y_pred_labels)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
