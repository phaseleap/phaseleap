# phaseleap/metrics.py
import numpy as np
from scipy.stats import entropy as shannon_entropy
import tensorflow as tf

def calculate_coherence(model):
    total_similarity = 0
    layers_considered = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()[0]
            norm = np.linalg.norm(weights)
            if norm != 0:
                similarity = np.sum(weights) / norm
                total_similarity += similarity
                layers_considered += 1
    return total_similarity / (layers_considered + 1e-5)

def calculate_entropy(model, x_sample):
    activations = []
    sample = x_sample[:10]  # Use consistent sample size

    # Precompile predictions
    _ = model.predict(sample)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
            layer_output = intermediate_model.predict(sample)
            activations.append(layer_output)

    flat_activations = np.concatenate([act.flatten() for act in activations])
    hist, _ = np.histogram(flat_activations, bins=50, density=True)
    hist = hist[hist > 0]
    return shannon_entropy(hist)

