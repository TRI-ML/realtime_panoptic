import copy

import cv2
import numpy as np
import torch

DETECTRON_PALETTE = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3) * 255


def visualize_segmentation_image(predictions, original_image, colormap, fade_weight=0.5):
    """Log a single segmentation result for visualization using a colormap.
    
    Overlays predicted classes on top of raw RGB image if given.

    Parameters:
    -----------
    predictions: torch.cuda.LongTensor
        Per-pixel predicted class ID's for a single input image
        Shape: (H, W)

    original_image: np.array
        HxWx3 original image. or None

    colormap: np.array
        (N+1)x3 array colormap,where N+1 equals to the number of classes.

    fade_weight: float, default: 0.8
        Visualization is fade_weight * original_image + (1 - fade_weight) * predictions

    Returns:
    --------
    visualized_image: np.array
        Semantic semantic visualization color coded by classes.
        The visualization will be overlaid on a the RGB image if given. 
    """

    # ``original_image`` has shape (H, W,3)
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
    original_image_height, original_image_width,_ = original_image.shape

    # Grab colormap from dataset for the given number of segmentation classes
    # (uses black for the IGNORE class)
    # ``colormap`` has shape (num_classes + 1, 3)

    # Color per-pixel predictions using the generated color map
    # ``colored_predictions_numpy`` has shape (H, W, 3)
    predictions_numpy = predictions.cpu().numpy().astype('uint8')
    colored_predictions_numpy = colormap[predictions_numpy.flatten()]
    colored_predictions_numpy = colored_predictions_numpy.reshape(original_image_height, original_image_width, 3)

    # Overlay images and predictions
    overlaid_predictions = original_image * fade_weight + colored_predictions_numpy * (1 - fade_weight)

    visualized_image = overlaid_predictions.astype('uint8')
    return visualized_image

def random_color(base, max_dist=30):
    """Generate random color close to a given base color.

    Parameters:
    -----------
    base: array_like
        Base color for random color generation

    max_dist: int
        Max distance from generated color to base color on all RGB axis.
    
    Returns:
    --------
    random_color: tuple
        3 channel random color around the given base color.
    """
    base = np.array(base)
    new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    return tuple(np.maximum(0, np.minimum(255, new_color)))

def draw_mask(im, mask, alpha=0.5, color=None):
    """Overlay a mask on top of the image.

    Parameters:
    -----------
    im: array_like
        A 3-channel uint8 image

    mask: array_like
        A binary 1-channel image of the same size

    color: bool
        If None, will choose automatically

    alpha: float
        mask intensity

    Returns:
    --------
    im: np.array
        Image overlaid by masks.

    color: list
        Color used for masks. 
    """
    if color is None:
        color = DETECTRON_PALETTE[np.random.choice(len(DETECTRON_PALETTE))][::-1]
    color = np.asarray(color, dtype=np.int64)
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2), im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im, color.tolist()

def visualize_detection_image(predictions, original_image, label_id_to_names, fade_weight=0.8):
    """Log a single detection result for visualization.
    
    Overlays predicted classes on top of raw RGB image.

    Parameters:
    -----------
    predictions: torch.cuda.LongTensor
        Per-pixel predicted class ID's for a single input image
        Shape: (H, W)

    original_image: np.array
        HxWx3 original image. or None

    label_id_to_names: list
        list of class names for instance labels

    fade_weight: float, default: 0.8
        Visualization is fade_weight * original_image + (1 - fade_weight) * predictions

    Returns:
    -------
    visualized_image: np.array
        Visualized image with detection results.
    """

    # Load raw image using provided dataset and index
    # ``images_numpy`` has shape (H, W, 3)
    # ``images_numpy`` has shape (H, W,3)
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
    original_image_height, original_image_width,_ = original_image.shape

    # overlay_boxes
    visualized_image = copy.copy(np.array(original_image))

    labels = predictions.get_field("labels").to("cpu")
    boxes = predictions.bbox

    dtype = labels.dtype
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1]).to(dtype)
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    masks = None
    if predictions.has_field("mask"):
        masks = predictions.get_field("mask")
    else:
        masks = [None] * len(boxes)
    # overlay_class_names_and_score
    if predictions.has_field("scores"):
        scores = predictions.get_field("scores").tolist()
    else:
        scores = [1.0] * len(boxes)
    # predicted label starts from 1 as 0 is reserved for background.
    label_names = [label_id_to_names[i-1] for i in labels.tolist()]

    text_template = "{}: {:.2f}"

    for box, color, score, mask, label in zip(boxes, colors, scores, masks, label_names):
        if score < 0.5:
            continue
        box = box.to(torch.int64)
        color = random_color(color)
        color = tuple(map(int, color))

        if mask is not None:
            thresh = (mask > 0.5).cpu().numpy().astype('uint8')
            visualized_image, color = draw_mask(visualized_image, thresh)

        x, y = box[:2]
        s = text_template.format(label, score)
        cv2.putText(visualized_image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        visualized_image = cv2.rectangle(visualized_image, tuple(top_left), tuple(bottom_right), tuple(color), 1)
    return visualized_image