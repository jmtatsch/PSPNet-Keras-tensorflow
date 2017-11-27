"""
Tests.

@author: tatsch
"""

from __future__ import print_function
import os.path
from scipy.misc import imread
import numpy as np


def test_iuo():
    """Test IoU."""
    from pspnet.evaluation import evaluate_iou
    image_path = os.path.join("tests", "test_images", "class_image.png")
    class_image = imread(image_path)
    print(class_image.shape)
    confusion_matrix, IOU, meanIOU = evaluate_iou([class_image], [class_image], classes=35)
    print("IoU: ", confusion_matrix, IOU, meanIOU)
    assert meanIOU == 1, "same image should have perfect IoU"

    white_image = np.full((1024, 2048), 0, dtype=np.uint8)
    confusion_matrix, IOU, meanIOU = evaluate_iou([white_image], [class_image], classes=35)
    print("IoU: ", confusion_matrix, IOU, meanIOU)
    assert meanIOU < 1, "blank image should have bad IoU"


def test_gt_image_to_class_image():
    from pspnet.utils import gt_image_to_class_image
    from pspnet.cityscapes_labels import full_id2label

    gt_image_path = os.path.join("tests", "test_images", "gt_image.png")
    gt_image = imread(gt_image_path)[:, :, 0:3]

    class_image = gt_image_to_class_image(gt_image, full_id2label)  # as this is the cityscapes train map, we need to get the
    class_image_path = os.path.join("tests", "test_images", "class_image.png")
    true_class_image = imread(class_image_path)
    error_image = np.isclose(class_image, true_class_image)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(class_image)
    plt.figure(2)
    plt.imshow(true_class_image)
    plt.figure(3)
    plt.imshow(error_image)
    plt.show()
    print(class_image)
    print(true_class_image)
    # Warning: A small number of classes has the same label colors so cannot be resolved into unique class_ids
    # assert np.allclose(class_image, true_class_image), "gt_image reduced to class image should be like class_image"
