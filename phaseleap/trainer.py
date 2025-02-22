# phaseleap/trainer.py
import tensorflow as tf
import numpy as np
from phaseleap.metrics import calculate_coherence, calculate_entropy
from phaseleap.utils import trigger_phase_shift

def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_phase_shift_model(model, x_train, y_train, x_test, y_test, epochs=10):
    coherence_history, entropy_history, accuracy_history = [], [], []
    for epoch in range(epochs):
        model.fit(x_train, y_train, epochs=1, verbose=1)
        coh = calculate_coherence(model)
        ent = calculate_entropy(model, x_train)
        coherence_history.append(coh)
        entropy_history.append(ent)
        acc = model.evaluate(x_test, y_test, verbose=0)[1]
        accuracy_history.append(acc)

        # Adaptive threshold logic
        coh_window = coherence_history[-10:] if len(coherence_history) >= 10 else coherence_history
        ent_window = entropy_history[-10:] if len(entropy_history) >= 10 else entropy_history
        coh_mean, coh_std = np.mean(coh_window), np.std(coh_window)
        ent_mean, ent_std = np.mean(ent_window), np.std(ent_window)

        if coh < coh_mean - 0.5 * coh_std or ent > ent_mean + 0.5 * ent_std:
            model = trigger_phase_shift(model)
    return model, coherence_history, entropy_history, accuracy_history
