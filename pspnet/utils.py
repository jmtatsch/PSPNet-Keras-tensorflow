from __future__ import print_function
import colorsys
import numpy as np
from keras.models import Model


def preprocess_image(img, mean=np.array([[[123.68, 116.779, 103.939]]])):  # mean in rgb order
    """Preprocess an image as input."""
    float_img = img.astype('float16')
    centered_image = float_img - mean
    bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
    input_data = bgr_image[np.newaxis, :, :, :]  # Append sample dimension for keras
    return input_data


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for i in range(-1, 256):  # go through all possible classes and color their regions at once
        try:
            cl = class_id_to_rgb_map[i]
            colored_image[class_id_image[:, :] == i] = cl.color
        except KeyError as key_error:
            print("Warning: could not resolve color of classid %s" % key_error)
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
            print("Warning: could not resolve classid %s" % key_error)
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


def download_weights(name):
    """Download Keras weights from Dropbox."""
    print("Downloading Keras weights from Dropbox ...")
    link_dict = {'pspnet50_ade20k.h5': 'https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1',
                 'pspnet50_ade20k.json': 'https://www.dropbox.com/s/v41lvku2lx7lh6m/pspnet50_ade20k.json?dl=1',
                 'pspnet101_cityscapes.h5': 'https://www.dropbox.com/s/c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1',
                 'pspnet101_cityscapes.json': 'https://www.dropbox.com/s/fswowe8e3o14tdm/pspnet101_cityscapes.json?dl=1',
                 'pspnet101_voc2012.h5': 'https://www.dropbox.com/s/uvqj2cjo4b9c5wg/pspnet101_voc2012.h5?dl=1',
                 'pspnet101_voc2012.json': 'https://www.dropbox.com/s/rr5taqu19f5fuzy/pspnet101_voc2012.json?dl=1'}

    for key in link_dict:
        if name in key:
            url = link_dict[key]
            print("Downloading %s from %s" % (key, url))
            response = requests.get(url)
            with open(join("..", "weights", "keras", key), 'wb') as f:
                f.write(response.content)


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
