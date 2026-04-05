import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("Using TensorFlow:", tf.__version__)

# 2. Load data
train_df = pd.read_csv("dataset/mnist_train.csv")
test_df  = pd.read_csv("dataset/mnist_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:",  test_df.shape)
print("First 5 records:\n", train_df.head())

print("train_df.columns:", train_df.columns.tolist())
print("test_df.columns:",  test_df.columns.tolist())


# 3. Prepare X_train, y_train
X = train_df.iloc[:, 1:].values / 255.0  # all pixel columns
y = train_df.iloc[:, 0].values           # label column

X = X.reshape(-1, 28, 28, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

y_train = to_categorical(y_train, 10)
y_val   = to_categorical(y_val, 10)

print("Original X shape:", X.shape)
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# Prepare test set – drop label column
X_test = test_df.iloc[:, 1:].values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
print("X_test:", X_test.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,        # random rotations in [-10, 10] degrees
    width_shift_range=0.1,    # random horizontal shifts
    height_shift_range=0.1,   # random vertical shifts
    zoom_range=0.1            # random zoom in/out
    # Do NOT use horizontal_flip for digits (6 ↔ 9 ambiguity)
)

datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

batch_size = 128
epochs = 10

with tf.device('/GPU:0'):
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    print(history.history.keys())

plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predict on validation set
y_val_prob = model.predict(X_val)
y_val_pred = np.argmax(y_val_prob, axis=1)
y_val_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_val_true, y_val_pred)
print("Confusion matrix shape:", cm.shape)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation set")
plt.show()

preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(X_test) + 1),
    "Label":   y_pred
})

submission.to_csv("submission.csv", index=False)
print(submission.head())