import hydra
from pathlib import Path
from tensorflow.keras import datasets
import tensorflow as tf
import importlib
import os
import sys

import sys

sys.path.append("../../../seminars")

# --------------------------------------------------------------------------------- #
#          Script for simple cnn cifar10 train                                      #
#          main config: l5_visualization/configs/hydra/cifar10_train_config.yaml    #
#          usage: > cd l5_visualization                                             #
#                 > python launch/cifar10_train.py                                  #
#                                                                                   #
#          trained model will be saved in  l5_visualization/outputs/cifar10_train   #
# --------------------------------------------------------------------------------- #


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


@hydra.main(config_path="../configs/hydra", config_name=Path(__file__).stem + "_config")
def run(cfg):
    # load data
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # create model

    model = get_class(cfg.model.module_name, cfg.model.class_name)(**cfg.model.args)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.getcwd(),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    # train
    history = model.fit(
        train_images,
        train_labels,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(test_images, test_labels),
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    run()
