import tensorflow as tf
import numpy as np

import cv2
import IPython.display as display
import PIL.Image


def generate_all_activations():
    pass


class ActivationOptimizer(tf.Module):
    def __init__(
        self,
        model,
        layer_name: str,
        activation_index: tuple,
        steps: int,
        step_size: float,
        reg_coef: float,
    ):
        self.steps = tf.constant(steps)
        self.reg_coef = reg_coef
        self.step_size = tf.convert_to_tensor(step_size)
        self.model = model
        self.model_slice = tf.keras.Model(
            inputs=self.model.input, outputs=[self.model.get_layer(layer_name).output]
        )
        self.activation_index = activation_index

    def loss_func(self, image):
        image_batch = tf.expand_dims(image, axis=0)
        layer_activations = self.model_slice(image_batch)
        return tf.reduce_sum(
            layer_activations[0][self.activation_index]
        ) - self.reg_coef * tf.image.total_variation(image)

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),)
    )
    def __call__(self, image):
        loss = tf.constant(0.0)
        for _ in tf.range(self.steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `image`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(image)
                loss = self.loss_func(image)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, image)
            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            image = image + gradients * self.step_size
            image = tf.clip_by_value(image, -1, 1)

        return loss, image


class DeepDream(tf.Module):
    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, image, steps, step_size):
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `image`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(image)
                loss = self.loss_func(image, self.model)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, image)
            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            image = image + gradients * step_size
            image = tf.clip_by_value(image, -1, 1)

        return loss, image


class DeepResp:
    def __init__(self, model, loss_func, steps, step_size):
        self.steps = steps
        self.step_size = step_size
        self.deepdream = DeepDream(model, loss_func)

    def deprocess(self, image):
        image = 255 * (image)
        return tf.cast(image, tf.uint8)

    # Display an image
    def show(self, image):
        display.display(PIL.Image.fromarray(cv2.resize(np.array(image), (256, 256))))

    def run_deep_dream_simple(self, image):
        # Convert from uint8 to the range expected by the model.
        step_size = tf.convert_to_tensor(self.step_size)
        steps_remaining = self.steps
        step = 0
        while steps_remaining:
            if steps_remaining > 50:
                run_steps = tf.constant(50)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, image = self.deepdream(image, run_steps, tf.constant(step_size))
            display.clear_output(wait=True)
            self.show(self.deprocess(image))
            print("Step {}, loss {}".format(step, loss))

        result = self.deprocess(image)
        display.clear_output(wait=True)
        self.show(result)

        return result

    def __call__(self, image):
        return self.run_deep_dream_simple(image)
