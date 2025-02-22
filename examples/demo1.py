# examples/demo.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from phaseleap import build_model, train_phase_shift_model, calculate_coherence, calculate_entropy

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 1️⃣ Build Model
model = build_model(input_shape=(28, 28, 1), num_classes=10)

# 2️⃣ Train with Phase Shift Learning
model, coh_history, ent_history, acc_history = train_phase_shift_model(
    model, x_train, y_train, x_test, y_test, epochs=10
)

# 3️⃣ Visualize Coherence vs Entropy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(coh_history, ent_history, marker='o')
plt.xlabel("Coherence")
plt.ylabel("Entropy")
plt.title("Phase Portrait: Coherence vs Entropy")

# 4️⃣ Visualize Accuracy Over Time
plt.subplot(1, 2, 2)
plt.plot(acc_history, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy with Phase Shifts")
plt.legend()

plt.show()
