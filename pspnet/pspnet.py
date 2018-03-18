#!/usr/bin/env python
"""
A Keras/Tensorflow implementation of Pyramid Scene Parsing Networks.

Original paper & code published by Hengshuang Zhao et al. (2017)
"""
from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile, isdir
from os import environ, walk
from math import ceil
import argparse
import glob
import fnmatch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
from layers_builder import build_pspnet
from utils import download_weights, download_npy_weights, preprocess_image, color_class_image, gt_image_to_class_image
from evaluation import evaluate_iou

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

__author__ = "Vlad Kryvoruchko, Chaoyue Wang, Jeffrey Hu & Julian Tatsch"


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        """Instanciate a PSPNet."""
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        json_path = join("..", "weights", "keras", weights + ".json")
        h5_path = join("..", "weights", "keras", weights + ".h5")

        if not isfile(json_path) and not isfile(h5_path):
            download_weights(weights)

        if isfile(json_path) and isfile(h5_path):
            print("Keras model & weights found, loading...")
            with open(json_path, 'r') as file_handle:
                try:
                    self.model = model_from_json(file_handle.read())
                except ValueError as err:  # bad marshal data error when loading py2 model in py3 an vice versa
                    # https://github.com/fchollet/keras/issues/7440
                    print("Couldn't import model from json because it was build using a different python version: %s" % err)
                    print("Rebuilding pspnet model ...")
                    self.model = build_pspnet(nb_classes=nb_classes,
                                              resnet_layers=resnet_layers,
                                              input_shape=self.input_shape)
                    print("Saving pspnet to disk ...")
                    json_string = self.model.to_json()
                    with open(json_path, 'w') as file_handle:
                        file_handle.write(json_string)
                except SystemError as err:  # bad marshal data error when loading py3.5 model in py3.6
                    print("Couldn't import model from json because it was build using a different python version: %s" % err)
                    print("Converting pspnet model from npy")
                    self.model = build_pspnet(nb_classes=nb_classes,
                                              resnet_layers=resnet_layers,
                                              input_shape=self.input_shape)
                    self.set_npy_weights(weights)

            self.model.load_weights(h5_path)
        else:
            print("No Keras model & weights found, import from npy weights.")
            self.model = build_pspnet(nb_classes=nb_classes,
                                      resnet_layers=resnet_layers,
                                      input_shape=self.input_shape)
            self.set_npy_weights(weights)

    def predict(self, img, flip_evaluation):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]
        if img.shape[0:2] != self.input_shape:
            print("Input %s not fitting for network size %s, resizing. You may want to try sliding prediction for better results." % (img.shape[0:2], self.input_shape))
            img = misc.imresize(img, self.input_shape)

        data = preprocess_image(img, mean=[[[174.08136209, 163.97867657, 138.72837669]]])
        # debug(self.model, input_data)
        if flip_evaluation:
            input_with_flipped = np.array([data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = prediction.shape[:2]
            prediction = ndimage.zoom(prediction, (1.*h_ori/h, 1.*w_ori/w, 1.),
                                      order=1, prefilter=False)
        return prediction

    def set_npy_weights(self, weights_path):
        """Set weights from the intermediary npy file."""
        npy_weights_path = join("..", "weights", "npy", weights_path + ".npy")
        json_path = join("..", "weights", "keras", weights_path + ".json")
        h5_path = join("..", "weights", "keras", weights_path + ".h5")

        if not isfile(npy_weights_path):
            download_npy_weights(weights_path)

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding="latin1").item()

        whitelist = ["InputLayer", "Activation", "ZeroPadding2D", "Add", "MaxPooling2D", "AveragePooling2D", "Lambda", "Concatenate", "Dropout"]

        weights_set = 0
        for layer in self.model.layers:
            print("Processing %s" % layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name]['mean'].reshape(-1)
                variance = weights[layer.name]['variance'].reshape(-1)
                scale = weights[layer.name]['scale'].reshape(-1)
                offset = weights[layer.name]['offset'].reshape(-1)
                self.model.get_layer(layer.name).set_weights([scale, offset, mean, variance])
                weights_set += 1
            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name]['weights']
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception:
                    biases = weights[layer.name]['biases']
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                 biases])
                weights_set += 1
            elif layer.__class__.__name__ in whitelist:
                # print("Nothing to set in %s" % layer.__class__.__name__)
                pass
            else:
                print("Warning: Did not find weights for keras layer %s in numpy weights" % layer)

        print("Set a total of %i weights" % weights_set)

        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        """Instanciate a PSPNet50."""
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        """Instanciate a PSPNet101."""
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img


def produce_view(input_image, class_image, id2label, viewstyle):
    """Produce an image ready for plotting or saving."""
    view = None
    if viewstyle == 'original':
        view = input_image
    elif (viewstyle == 'predictions') or (viewstyle == 'overlay'):
        view = color_class_image(class_image, id2label)
        if viewstyle == 'overlay':
            view = (0.5 * view.astype(np.float32) + 0.5 * input_image.astype(np.float32)).astype(np.uint8)
    else:
        print("Unknown view style")
    return view


def visualize_prediction(input_image, class_scores, id2label):
    """Visualize prediction in faux colors."""
    class_image = np.argmax(class_scores, axis=2)
    fig = plt.figure()
    axis = fig.add_subplot(111)

    def button_handler(viewstyle):
        axis.imshow(produce_view(input_image, class_image, id2label, viewstyle))
        plt.draw()

    # plt.subplots_adjust(left=0.3)
    rax = plt.axes([0.4, 0.05, 0.2, 0.15])
    radio_buttons = RadioButtons(rax, ('original', 'overlay', 'predictions'))
    radio_buttons.on_clicked(button_handler)

    # image = produce_view(input_image, class_image, 'overlay')
    # axis.imshow(image)
    button_handler('original')
    axis.set_axis_off()
    # overwrite the status bar with class information
    axis.format_coord = lambda x, y: id2label[class_image[int(y), int(x)]].name
    plt.show()


def show_class_heatmap(class_scores, class_name):
    """Show a heatmap with the probabilities of a certain class."""
    try:
        class_id = name2label[class_name].id
        class_heatmap = class_scores[:, :, class_id]
        plt.axis('off')
        plt.imshow(class_heatmap, cmap='coolwarm')
        plt.show()
    except KeyError as err:
        print("Could not find index for %s because of %s" % (class_name, err))


def show_class_heatmaps(class_scores):
    """
    Show heatmap with the probabilities of a certain class.

    Cycle through with lef and right arrow keys.
    """
    show_class_heatmaps.curr_index = 0

    def key_event(event):
        """Handle forward & backward arrow key presses."""
        if event.key == "right":
            show_class_heatmaps.curr_index += 1
        elif event.key == "left":
            show_class_heatmaps.curr_index -= 1
        else:
            return
        show_class_heatmaps.curr_index = show_class_heatmaps.curr_index % class_scores.shape[2]

        axis.cla()
        class_heatmap = class_scores[:, :, show_class_heatmaps.curr_index]
        axis.imshow(class_heatmap, cmap='coolwarm')
        axis.set_axis_off()
        fig.canvas.set_window_title(id2label[show_class_heatmaps.curr_index].name)
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    fig.canvas.set_window_title(id2label[show_class_heatmaps.curr_index].name)
    axis = fig.add_subplot(111)
    class_heatmap = class_scores[:, :, show_class_heatmaps.curr_index]
    axis.imshow(class_heatmap, cmap='coolwarm')
    axis.set_axis_off()
    plt.show()


def predict_sliding(full_image, net, flip_evaluation):
    """
    Predict on tiles of exactly the network input shape.

    This way nothing gets squeezed.
    """
    tile_size = net.input_shape
    classes = net.model.outputs[0].shape[3]
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = max(int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1), 1)  # strided convolution formula
    tile_cols = max(int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1), 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = full_image[y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction = net.predict(padded_img, flip_evaluation)
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def predict_multi_scale(full_image, net, scales, sliding_evaluation, flip_evaluation):
    """Predict an image by looking at it with different scales."""
    classes = net.model.outputs[0].shape[3]
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    h_ori, w_ori = full_image.shape[:2]
    for scale in scales:
        print("Predicting image scaled by %f" % scale)
        scaled_img = misc.imresize(full_image, size=scale, interp="bilinear")
        if sliding_evaluation:
            scaled_probs = predict_sliding(scaled_img, net, flip_evaluation)
        else:
            scaled_probs = net.predict(scaled_img, flip_evaluation)
        # scale probs up to full size
        h, w = scaled_probs.shape[:2]
        probs = ndimage.zoom(scaled_probs, (1.*h_ori/h, 1.*w_ori/w, 1.),
                             order=1, prefilter=False)
        # visualize_prediction(probs)
        # integrate probs over all scales
        full_probs += probs
    full_probs /= len(scales)
    return full_probs


def trainid_to_class_image(trainid_image):
    """Inflate an image with trainId's into a full class image with class ids."""
    from cityscapesscripts.helpers.labels import trainId2label
    class_image = np.zeros(trainid_image.shape, np.uint8)
    try:
        for row in range(trainid_image.shape[0]):
            for col in range(trainid_image.shape[1]):
                class_image[row][col] = trainId2label[trainid_image[row][col]].id
    except Exception as ex:
        print("Unknown trainid : %s" % ex)

    return class_image


def find_matching_gt(gt_dir, image_name, model_name, verbose=False):
    """Find a matching ground truth in gt_dir for image_name."""
    if "cityscapes" in model_name:
        filter_string = image_name + "*labelIds.png"
    else:
        filter_string = image_name + "*.png"
    for root, __, files in walk(gt_dir):
        for filename in fnmatch.filter(files, filter_string):
            if verbose:
                print("Found matching groundtruth at: %s" % join(root, filename))
            return join(root, filename)


def complete_coarse_image(coarse_image, predicted_img):
    """Complete a coarsely labeld cityscapes image with predictions."""
    mask_indices = coarse_image == 0  # complete everywhere where coarse_image is 0
    coarse_image[mask_indices] = predicted_img[mask_indices]
    return coarse_image


def main():
    """Run when running this module as the primary one."""
    EVALUATION_SCALES = [1.0]  # must be all floats!

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-i', '--input_path', type=str, default='../example_images',
                        help='Path to the input images')
    parser.add_argument('-o', '--output_path', type=str, default='../example_results',
                        help='Path to output')
    parser.add_argument('-g', '--groundtruth_path', type=str, default='../example_groundtruth',
                        help='Path to groundtruth')
    parser.add_argument('--id', default="0")
    parser.add_argument('-s', '--sliding', action='store_true', default=True,
                        help="Whether the network should be slided over the original image for prediction.")
    parser.add_argument('-f', '--flip', action='store_true', default=True,
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-ms', '--multi_scale', action='store_true',
                        help="Whether the network should predict on multiple scales.")
    parser.add_argument('-hm', '--heat_maps', action='store_true',
                        help="Whether the network should diplay heatmaps.")
    parser.add_argument('-v', '--vis', action='store_true',
                        help="Whether an interactive plot should be diplayed.")
    parser.add_argument('-cci', '--complete_coarse_image', action='store_true',
                        help="Whether a coarse imae should be completed with predictions.")
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help="Whether an evaluation against groundtruth should be attempted.")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        import os
        cwd = os.getcwd()
        print("Running in %s" % cwd)

        image_paths = []
        if isfile(args.input_path):
            image_paths.append(args.input_path)
        elif isdir(args.input_path):
            file_types = ('png', 'jpg')
            for file_type in file_types:
                image_paths.extend(glob.glob(join(args.input_path + '/**/*.' + file_type), recursive=True))
            image_paths = sorted(image_paths)
        # print(image_paths)

        if "pspnet50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=args.model)
            if "ade20k" in args.model:
                from ade20k_labels import id2label, name2label

        elif "pspnet101" in args.model:
            if "cityscapes" in args.model:
                pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                   weights=args.model)
                from cityscapes_labels import id2label, name2label
            if "voc2012" in args.model:
                pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                   weights=args.model)
                from pascal_voc_labels import id2label, name2label

        else:
            print("Network architecture not implemented.")

        if args.multi_scale:
            EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # original implementation, must be all floats!

        for image_path in image_paths:
            image_name, ext = splitext(os.path.basename(image_path))
            image_name = image_name.replace('_leftImg8bit', '')  # strip leftImg8bit tag for gt matching and producting groundtruth

            print("Predicting image name: %s" % (image_name + ext))
            img = misc.imread(image_path)
            class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, args.sliding, args.flip)
            if args.heat_maps:
                # show_class_heatmap(class_scores, 'person')
                show_class_heatmaps(class_scores)

            # visualize_prediction(img, class_scores, id2label)
            class_image = np.argmax(class_scores, axis=2)

            output_path, _ = splitext(args.output_path)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = join(output_path, image_name)

            print("Writing results to %s" % (output_path + ext))

            confidence_map = np.max(class_scores, axis=2)  # probability of the most likely class, a vage measure of the networks confidence
            colored_class_image = color_class_image(class_image, id2label)

            # colored_class_image is [0.0-1.0] img is [0-255]
            alpha_blended = 0.5 * colored_class_image + 0.5 * img
            if "cityscapes" in args.model:
                class_image = trainid_to_class_image(class_image)
            misc.imsave(output_path + "_gtFine_labelIds" + ext, class_image)
            misc.imsave(output_path + "_seg" + ext, colored_class_image)
            misc.imsave(output_path + "_probs" + ext, confidence_map)
            misc.imsave(output_path + "_seg_blended" + ext, alpha_blended)

            gt_path = find_matching_gt(args.groundtruth_path, image_name, args.model, verbose=True)
            if gt_path is not None:
                if args.complete_coarse_image:  # only for cityscapes
                    try:
                        coarse_image = misc.imread(gt_path)
                        class_image = complete_coarse_image(coarse_image, class_image)
                        misc.imsave(output_path + "_gtFine_labelIds" + ext, class_image)
                    except AttributeError as err:
                        print("Warning: Could not read groundtruth: %s" % err)

                if args.evaluate:
                    if "cityscapes" in args.model:
                        evaluate_iou([class_image], [misc.imread(gt_path)], classes=35)
                    else:
                        # gt_image to class image
                        gt_image = misc.imread(gt_path)
                        gt_class_image = gt_image_to_class_image(gt_image, id2label)
                        evaluate_iou([class_image], [gt_class_image], classes=pspnet.nb_classes)
            else:
                print("Could not find groundtruth for %s" % image_name)


if __name__ == "__main__":
    main()

