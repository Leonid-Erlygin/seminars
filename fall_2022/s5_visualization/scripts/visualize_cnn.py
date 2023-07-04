import sys

sys.path.append(
    "../../../seminars/s5_visualization/vis_tools/pytorch_cnn_visualizations/src"
)


from matplotlib import pyplot as plt
import numpy as np
import torch


from nn_interpretability.interpretation.backprop.guided_backprop import GuidedBackprop
from nn_interpretability.visualization.rgb_visualizer import RGBVisualizer
from nn_interpretability.interpretation.backprop.vanilla_backprop import VanillaBackprop
from nn_interpretability.interpretation.backprop.guided_backprop import GuidedBackprop
from nn_interpretability.interpretation.backprop.integrated_grad import IntegratedGrad


from layercam import LayerCam
from gradcam import GradCam
from misc_functions import apply_colormap_on_image, apply_heatmap, convert_to_grayscale
from LRP import LRP
from matplotlib.colors import ListedColormap

from PIL import Image
import cv2
from pathlib import Path
from itertools import product


def apply_heatmap(R):
    """
    Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial

    This is (so far) only used for LRP
    """
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    return my_cmap, b


def get_explanations(
    model,
    image_transformed: torch.tensor,
    image_unchanged: Image,
    image_category_name: str,
    image_category: int,
    save_path: Path,
):
    INTEGRATED_GRAD_STEPS = 10
    image_shape = image_unchanged.size

    num_row = 5
    num_col = 3

    fig, axs = plt.subplots(num_row, num_col, figsize=(10, 15))
    # set ticks to None
    for i, j in product(range(num_row), range(num_col)):
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

    # draw original image
    axs[0, 0].imshow(image_unchanged)
    axs[0, 0].set_title(f"input image: {image_category_name}")

    # gradcam on image

    grad_cam_extractor = GradCam(model, target_layer=28)
    grad_cam = cv2.resize(
        grad_cam_extractor.generate_cam(image_transformed, target_class=image_category),
        image_shape,
    )

    heatmap_grad_cam, heatmap_on_image_grad_cam = apply_colormap_on_image(
        image_unchanged, grad_cam, "hsv"
    )

    axs[0, 1].imshow(heatmap_on_image_grad_cam)
    axs[0, 1].set_title("GRAD-CAM on image")

    # gradcam
    axs[0, 2].imshow(heatmap_grad_cam)
    axs[0, 2].set_title("GRAD-CAM")

    # Vallina Backpropagation
    interpretor = VanillaBackprop(model, [], None)
    endpoint = interpretor.interpret(image_transformed, target_class=image_category)
    gray = convert_to_grayscale(endpoint.detach().numpy()[0])
    gray = gray - gray.min()
    gray /= gray.max()
    backprop_image = cv2.resize(gray[0], image_shape)
    backprop_image = (backprop_image * 255).astype("uint8")
    axs[1, 0].imshow(backprop_image, interpolation="nearest")
    axs[1, 0].set_title("Vallina Backpropagation")

    # Guided Backpropagation
    interpretor = GuidedBackprop(model, [], None)
    endpoint = interpretor.interpret(image_transformed, target_class=image_category)
    guided_backprop = RGBVisualizer.postprocess(endpoint)
    guided_backprop = cv2.resize(guided_backprop, image_shape)

    guided_backprop = (guided_backprop * 255).astype("uint8")
    axs[1, 1].imshow(guided_backprop, interpolation="nearest")
    axs[1, 1].set_title("Guided Backpropagation ")

    # Integrated Gradients
    baseline = torch.zeros_like(image_transformed)
    interpretor = IntegratedGrad(model, [], None, baseline, steps=INTEGRATED_GRAD_STEPS)
    endpoint = interpretor.interpret(image_transformed)
    gray = convert_to_grayscale(endpoint.detach().numpy()[0])
    gray = gray - gray.min()
    gray /= gray.max()
    int_grad_image = cv2.resize(gray[0], image_shape)

    int_grad_image = (int_grad_image * 255).astype("uint8")
    axs[1, 2].imshow(int_grad_image, interpolation="nearest")
    axs[1, 2].set_title("Integrated Gradients")

    # layer CAM. Layer 30

    layer_cam_extractor = LayerCam(model, target_layer=30)
    layer_cam = cv2.resize(
        layer_cam_extractor.generate_cam(
            image_transformed, target_class=image_category
        ),
        image_shape,
    )

    heatmap_layer_cam, heatmap_on_image_layer_cam = apply_colormap_on_image(
        image_unchanged, layer_cam, "hsv"
    )

    axs[2, 0].imshow(heatmap_on_image_layer_cam)
    axs[2, 0].set_title("Layer-CAM(Layer 30) on image")

    axs[3, 0].imshow(heatmap_layer_cam)
    axs[3, 0].set_title("Layer-CAM(Layer 30)")

    # layer CAM. Layer 23

    layer_cam_extractor = LayerCam(model, target_layer=23)
    layer_cam = cv2.resize(
        layer_cam_extractor.generate_cam(
            image_transformed, target_class=image_category
        ),
        image_shape,
    )

    heatmap_layer_cam, heatmap_on_image_layer_cam = apply_colormap_on_image(
        image_unchanged, layer_cam, "hsv"
    )

    axs[2, 1].imshow(heatmap_on_image_layer_cam)
    axs[2, 1].set_title("Layer-CAM(Layer 23) on image")

    axs[3, 1].imshow(heatmap_layer_cam)
    axs[3, 1].set_title("Layer-CAM(Layer 23)")

    # layer CAM. Layer 16

    layer_cam_extractor = LayerCam(model, target_layer=16)
    layer_cam = cv2.resize(
        layer_cam_extractor.generate_cam(
            image_transformed, target_class=image_category
        ),
        image_shape,
    )

    heatmap_layer_cam, heatmap_on_image_layer_cam = apply_colormap_on_image(
        image_unchanged, layer_cam, "hsv"
    )

    axs[2, 2].imshow(heatmap_on_image_layer_cam)
    axs[2, 2].set_title("Layer-CAM(Layer 16) on image")

    axs[3, 2].imshow(heatmap_layer_cam)
    axs[3, 2].set_title("Layer-CAM(Layer 16)")

    # LRP. Layer 1
    layer = 1
    layerwise_relevance = LRP(model)
    LRP_per_layer = layerwise_relevance.generate(
        image_transformed, target_class=image_category
    )

    # Convert the output nicely, selecting the first layer
    lrp_to_vis = np.array(LRP_per_layer[layer][0].cpu()).sum(axis=0)
    lrp_to_vis = np.array(
        Image.fromarray(lrp_to_vis).resize(
            (image_unchanged.size[0], image_unchanged.size[1]), Image.ANTIALIAS
        )
    )

    my_cmap, b = apply_heatmap(lrp_to_vis)
    axs[4, 0].imshow(lrp_to_vis, cmap=my_cmap, vmin=-b, vmax=b, interpolation="nearest")
    axs[4, 0].set_title(f"LRP(Layer {layer})")

    # LRP. Layer 7
    layer = 7
    layerwise_relevance = LRP(model)
    LRP_per_layer = layerwise_relevance.generate(
        image_transformed, target_class=image_category
    )

    # Convert the output nicely, selecting the first layer
    lrp_to_vis = np.array(LRP_per_layer[layer][0].cpu()).sum(axis=0)
    lrp_to_vis = np.array(
        Image.fromarray(lrp_to_vis).resize(
            (image_unchanged.size[0], image_unchanged.size[1]), Image.ANTIALIAS
        )
    )

    my_cmap, b = apply_heatmap(lrp_to_vis)
    axs[4, 1].imshow(lrp_to_vis, cmap=my_cmap, vmin=-b, vmax=b, interpolation="nearest")
    axs[4, 1].set_title(f"LRP(Layer {layer})")

    # LRP. Layer 16
    layer = 16
    layerwise_relevance = LRP(model)
    LRP_per_layer = layerwise_relevance.generate(
        image_transformed, target_class=image_category
    )

    # Convert the output nicely, selecting the first layer
    lrp_to_vis = np.array(LRP_per_layer[layer][0].cpu()).sum(axis=0)
    lrp_to_vis = np.array(
        Image.fromarray(lrp_to_vis).resize(
            (image_unchanged.size[0], image_unchanged.size[1]), Image.ANTIALIAS
        )
    )

    my_cmap, b = apply_heatmap(lrp_to_vis)
    axs[4, 2].imshow(lrp_to_vis, cmap=my_cmap, vmin=-b, vmax=b, interpolation="nearest")
    axs[4, 2].set_title(f"LRP(Layer {layer})")

    fig.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    plt.close(fig)
