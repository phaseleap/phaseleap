# phaseleap/utils.py
import numpy as np
import tensorflow as tf

def trigger_phase_shift(model):
    """Randomly perturb layer weights and adjust the learning rate."""
    print("🔥 Phase Shift Triggered!")

    for layer in model.layers:
        if isinstance(
            layer,
            (
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D,
                tf.keras.layers.LSTM,
                tf.keras.layers.GRU,
                tf.keras.layers.SimpleRNN,
            ),
        ):
            layer_weights = layer.get_weights()
            if not layer_weights:
                continue

            # Determine which weights correspond to the bias term
            weight_indices = range(len(layer_weights))
            if getattr(layer, "use_bias", False):
                weight_indices = range(len(layer_weights) - 1)

            for idx in weight_indices:
                w = layer_weights[idx]
                weight_variance = np.var(w)

                if weight_variance > 0.05:
                    layer_weights[idx] = np.random.normal(0, 0.1, w.shape)
                else:
                    perturbation = np.random.normal(0, 0.02, w.shape)
                    layer_weights[idx] = w + perturbation

            layer.set_weights(layer_weights)

    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    new_lr = current_lr * np.random.uniform(0.5, 1.5)

    if isinstance(model.optimizer.learning_rate, tf.Variable):
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
    else:
        model.optimizer.learning_rate = new_lr

    print(f"🔄 New Learning Rate: {new_lr:.5f}")
    return model
