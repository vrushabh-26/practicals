# ---------- Import required libraries ----------
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ---------- 1. Generate Training Data ----------
np.random.seed(42)
num_samples = 5000
latent_dim = 5

# Generate latent noise samples (input)
Z = np.random.normal(0, 1, (num_samples, latent_dim))

# Define target output: sin(z0) and cos(z1)
Y = np.zeros((num_samples, 2))
Y[:, 0] = np.sin(Z[:, 0])
Y[:, 1] = np.cos(Z[:, 1])

# ---------- 2. Build the Generative Multi-Layer Network Model ----------
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(latent_dim,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.summary()

# ---------- 3. Train the Model ----------
history = model.fit(Z, Y, epochs=100, batch_size=32, verbose=1)

# ---------- 4. Generate New Samples Using the Trained Model ----------
Z_new = np.random.normal(0, 1, (1000, latent_dim))
generated_outputs = model.predict(Z_new)

# ---------- 5. Visualize the Generated Outputs ----------
plt.figure(figsize=(6, 6))
plt.scatter(generated_outputs[:, 0], generated_outputs[:, 1], alpha=0.6)
plt.title("Generated Outputs from Generative Multi-Layer Model")
plt.xlabel("Output Dimension 1 (sin)")
plt.ylabel("Output Dimension 2 (cos)")
plt.grid()
plt.savefig("experiment4_output.png", dpi=150, bbox_inches='tight')
plt.show()

# ---------- 6. Plot Training Loss ----------
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], color='blue')
plt.title("Training Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.savefig("experiment4_loss.png", dpi=150, bbox_inches='tight')
plt.show()

print("Done! Plots saved.")