# Experiment No. 6
# FAST Transfer Learning using MobileNetV2
# (Optimized alternative to VGG16 for faster execution)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report

# -----------------------------------------
# 1. LOAD AND PREPROCESS CIFAR-10
# -----------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reduce dataset size for faster training
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# Resize images to 64x64
x_train_resized = tf.image.resize(x_train, (64, 64))
x_test_resized = tf.image.resize(x_test, (64, 64))

# Preprocess for MobileNetV2
x_train_resized = preprocess_input(x_train_resized)
x_test_resized = preprocess_input(x_test_resized)

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# -----------------------------------------
# 2. LOAD PRE-TRAINED MODEL
# -----------------------------------------

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(64, 64, 3)
)

# Freeze base model
base_model.trainable = False

# -----------------------------------------
# 3. BUILD CUSTOM MODEL
# -----------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.4)(x)

output = Dense(10, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# -----------------------------------------
# 4. COMPILE MODEL
# -----------------------------------------

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------

print("\nStarting Fast Transfer Learning...\n")

history = model.fit(
    x_train_resized,
    y_train_cat,
    epochs=3,                 # reduced epochs
    batch_size=128,           # faster batching
    validation_split=0.2,
    verbose=1
)

# -----------------------------------------
# 6. EVALUATE MODEL
# -----------------------------------------

print("\nEvaluating Model...\n")

test_loss, test_acc = model.evaluate(
    x_test_resized,
    y_test_cat,
    verbose=1
)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------------------
# 7. PREDICTIONS
# -----------------------------------------

y_pred_probs = model.predict(x_test_resized)

y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test.flatten(),
        y_pred,
        target_names=CLASS_NAMES
    )
)

# -----------------------------------------
# 8. VISUALIZE PREDICTIONS
# -----------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

fig.suptitle(
    "Fast Transfer Learning Predictions",
    fontsize=14
)

for i, ax in enumerate(axes.flat):

    image = (x_test[i] / 255.0)

    pred_label = CLASS_NAMES[y_pred[i]]

    actual_label = CLASS_NAMES[y_test[i][0]]

    color = 'green' if pred_label == actual_label else 'red'

    ax.imshow(image)

    ax.set_title(
        f"P: {pred_label}\nA: {actual_label}",
        color=color,
        fontsize=9
    )

    ax.axis('off')

plt.tight_layout()

plt.savefig("predictions.png", dpi=150)

plt.show()

# -----------------------------------------
# 9. PLOT ACCURACY & LOSS
# -----------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train Acc')

ax1.plot(history.history['val_accuracy'], label='Val Acc')

ax1.set_title("Accuracy over Epochs")

ax1.set_xlabel("Epoch")

ax1.set_ylabel("Accuracy")

ax1.legend()

ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Train Loss')

ax2.plot(history.history['val_loss'], label='Val Loss')

ax2.set_title("Loss over Epochs")

ax2.set_xlabel("Epoch")

ax2.set_ylabel("Loss")

ax2.legend()

ax2.grid(True)

plt.tight_layout()

plt.savefig("accuracy_loss.png", dpi=150)

plt.show()

print("\nTraining Complete!")