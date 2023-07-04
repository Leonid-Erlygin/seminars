from tensorflow import keras
from tensorflow.keras import layers


def simple_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, strides=1, padding="same")(inputs)
    x = layers.Conv2D(16, 3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Activation("relu")(x)
    outputs = layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)
