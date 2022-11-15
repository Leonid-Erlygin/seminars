import sys

sys.path.append("../../../seminars")

import hydra
from pathlib import Path
import hydra
from tensorflow import keras
from l5_visualization.scripts.optimizators import ActivationOptimizer
import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------------- #
#          Script for layer visualization                                           #
#          main config: l5_visualization/configs/hydra/optimize_layers_config.yaml  #
#          usage: > cd l5_visualization                                             #
#                 > python launch/optimize_layers.py                                #
#                                                                                   #
#          visualizations will be saved in l5_visualization/outputs/optimize_layers#
# --------------------------------------------------------------------------------- #


@hydra.main(config_path="../configs/hydra", config_name=Path(__file__).stem + "_config")
def run(cfg):
    model = keras.models.load_model(cfg.model_path)

    out_dir = Path("layers_vis")
    print(f"saving visualization to {str(out_dir.absolute())}")
    for layer_name in cfg.layers_to_optimize:
        layer = model.get_layer(layer_name)

        for filter_index in range(layer.filters):
            if cfg.optimize_center is False:
                activation_index = (slice(None), slice(None), filter_index)
            else:
                activation_index = (
                    layer.input.shape[1] // 2,
                    layer.input.shape[1] // 2,
                    filter_index,
                )
            ao = ActivationOptimizer(
                model=model,
                layer_name=layer.name,
                activation_index=activation_index,
                steps=cfg.steps,
                step_size=cfg.step_size,
                reg_coef=cfg.reg_coef,
            )
            for i in range(cfg.n_images_per_filter):
                random_image = tf.random.uniform(
                    [32, 32, 3],
                    minval=0,
                    maxval=None,
                    dtype=tf.dtypes.float32,
                    seed=None,
                    name=None,
                )
                loss, image_raw = ao(random_image)

                image = 255 * (image_raw)
                image = tf.cast(image, tf.uint8)
                image = np.array(image)
                # resize
                image = cv2.resize(image, cfg.resize_image_to)

                # save
                image_out_dir = out_dir / layer.name / f"filter_{filter_index}"
                image_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(image_out_dir / f"sample_{i}") + ".png",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                )


if __name__ == "__main__":
    run()
