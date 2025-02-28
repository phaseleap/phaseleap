# phaseleap/utils.py
import numpy as np
import tensorflow as tf

def trigger_phase_shift(model):
    print("🔥 Phase Shift Triggered!")
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN)):
            weights, biases = layer.get_weights()
            weight_variance = np.var(weights)

            if weight_variance > 0.05:
                new_weights = np.random.normal(0, 0.1, weights.shape)
                layer.set_weights([new_weights, biases])
            else:
                perturbation = np.random.normal(0, 0.02, weights.shape)
                layer.set_weights([weights + perturbation, biases])

    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    new_lr = current_lr * np.random.uniform(0.5, 1.5)

    if isinstance(model.optimizer.learning_rate, tf.Variable):
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
    else:
        model.optimizer.learning_rate = new_lr

    print(f"🔄 New Learning Rate: {new_lr:.5f}")
    return model
