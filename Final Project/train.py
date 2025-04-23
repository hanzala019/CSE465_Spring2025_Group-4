import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score




# Load the training, validation, and test sets
train_df = pd.read_csv("dataset/train_set.csv")
val_df = pd.read_csv("dataset/val_set.csv")
test_df = pd.read_csv("dataset/test_set.csv")

# Get all columns for x, y, z (ignore time columns)
feature_cols = [col for col in train_df.columns if col.startswith(('x', 'y', 'z'))]

def reshape_for_lstm(df):
    X = df[feature_cols].values
    # Reshape from (samples, 15) to (samples, 5 timesteps, 3 features)
    X = X.reshape((X.shape[0], 5, 3))
    return X

X_train = reshape_for_lstm(train_df)
X_val = reshape_for_lstm(val_df)
X_test = reshape_for_lstm(test_df)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['activity_id'])
y_val = label_encoder.transform(val_df['activity_id'])
y_test = label_encoder.transform(test_df['activity_id'])

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Model
model = Sequential()
model.add(LSTM(64, input_shape=(5, 3)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=20,
    batch_size=32
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.2f}")


model.save("lstm_model.h5")