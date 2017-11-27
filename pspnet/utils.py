from __future__ import print_function
import colorsys
import numpy as np
from keras.models import Model


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for i in range(-1, 256):  # go through possible classes and color their regions at once
        try:
            cl = class_id_to_rgb_map[i]
            colored_image[class_id_image[:, :] == i] = cl.color
        except KeyError as key_error:
            # print("Warning: could not resolve classid %s" % key_error)
            pass
    return colored_image


def gt_image_to_class_image(gt_image, class_id_to_rgb_map):
    """Map the rgb-color gt_image to a class image."""
    class_image = np.zeros((gt_image.shape[0], gt_image.shape[1]), np.uint8)
    for class_id in range(-1, 256):  # go through possible classes and color their regions at once
        try:
            class_color = list(class_id_to_rgb_map[class_id].color)
            # print("treating class %i i.e. color %s" % (class_id, class_color))
            class_image[np.where((gt_image == class_color).all(axis=2))] = class_id
        except KeyError as key_error:
            # print("Warning: could not resolve classid %s" % key_error)
            pass
    return class_image


def color_class_image(class_image, id2label):
    """Color classes according to their original colormap."""
    if id2label:
        colored_image = class_image_to_image(class_image, id2label)
        colored_image = class_image_to_image(class_image, id2label)
        colored_image = class_image_to_image(class_image, id2label)
    else:
        colored_image = add_color(class_image)
    return colored_image


def add_color(img):
    """Color classes a good distance away from each other."""
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in xrange(1, 151):
        img_color[img == i] = to_color(i)
    return img_color * 255  # is [0.0-1.0]  should be [0-255]


def to_color(category):
    """Map each category color a good distance from each other on the HSV color space."""
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v, 1, 1)


def debug(model, data):
    """Debug model by printing the activations in each layer."""
    names = [layer.name for layer in model.layers]
    for name in names[:]:
        print_activation(model, name, data)


def print_activation(model, layer_name, data):
    """Print the activations in each layer."""
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print(layer_name, array_to_str(io))


def array_to_str(a):
    """Dume activation parameters into a string."""
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a),
                                   np.max(a), np.mean(a))
