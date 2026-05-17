# Experiment No. 6
# Aim:
# To explore the working of a pre-trained model (VGG16)
# for outcome generation and use transfer learning
# to classify images from the CIFAR-10 dataset.

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
import seaborn as sns

# ---------------------------------------------------
# 1. Load and Preprocess CIFAR-10 Dataset
# ---------------------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize images to 48x48
x_train_resized = tf.image.resize(x_train, [48, 48]).numpy() / 255.0
x_test_resized = tf.image.resize(x_test, [48, 48]).numpy() / 255.0

# One-hot encode labels
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

# ---------------------------------------------------
# 2. Load Pre-trained VGG16 without Top Layers
# ---------------------------------------------------

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(48, 48, 3)
)

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# ---------------------------------------------------
# 3. Create Custom Classification Model
# ---------------------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(10, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.summary()

# ---------------------------------------------------
# 4. Compile and Train the Model
# ---------------------------------------------------

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting Training...\n")

history = model.fit(
    x_train_resized,
    y_train_cat,
    epochs=2,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# ---------------------------------------------------
# 5. Evaluate the Model
# ---------------------------------------------------

print("\nEvaluating Model...\n")

test_loss, test_acc = model.evaluate(
    x_test_resized,
    y_test_cat,
    verbose=0
)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(
    x_test_resized,
    batch_size=128
)

y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test.flatten(),
        y_pred,
        target_names=CLASS_NAMES
    )
)

# ---------------------------------------------------
# 6. Visualize Sample Predictions
# ---------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

fig.suptitle(
    "VGG16 Transfer Learning - Sample Predictions on CIFAR-10",
    fontsize=14
)

for i, ax in enumerate(axes.flat):

    ax.imshow(x_test_resized[i])

    pred_label = CLASS_NAMES[y_pred[i]]

    actual_label = CLASS_NAMES[y_test[i][0]]

    color = 'green' if pred_label == actual_label else 'red'

    ax.set_title(
        f"P: {pred_label}\nA: {actual_label}",
        color=color,
        fontsize=9
    )

    ax.axis('off')

plt.tight_layout()

plt.show()

# ---------------------------------------------------
# 7. Plot Training and Validation Accuracy/Loss
# ---------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy Plot
ax1.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

ax1.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

ax1.set_title('Accuracy over Epochs')

ax1.set_xlabel('Epoch')

ax1.set_ylabel('Accuracy')

ax1.legend()

ax1.grid(True)

# Loss Plot
ax2.plot(
    history.history['loss'],
    label='Train Loss'
)

ax2.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

ax2.set_title('Loss over Epochs')

ax2.set_xlabel('Epoch')

ax2.set_ylabel('Loss')

ax2.legend()

ax2.grid(True)

plt.tight_layout()

plt.show()