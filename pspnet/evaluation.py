#!/usr/bin/python
"""evaluate iou result with cityscapes tool."""
import numpy as np
from scipy import misc


def evaluate_img_lists(pred_imgs_paths, gt_imgs_paths):
    """Evaluate lists with image paths."""
    pred_list = []
    gt_list = []
    for pred_path, gt_path in zip(pred_imgs_paths, gt_imgs_paths):
        pred_list.append(misc.imread(pred_path))
        gt_list.append(misc.imread(gt_path))
    evaluate_iou(pred_list, gt_list)


def evaluate_iou(pred_imgs, gt_imgs, classes=35):
    """Evaluate a batch of predicted images against groundtruth."""
    confusion_matrix = np.zeros((classes, classes), dtype=float)
    for predicted, groundtruth in zip(pred_imgs, gt_imgs):
        flat_pred = np.ravel(predicted)
        print(groundtruth.shape, np.min(groundtruth), np.max(groundtruth))
        flat_gt = np.ravel(groundtruth)
        for pred_val, gt_val in zip(flat_pred, flat_gt):
            if gt_val == 0:  # groundtruth doesn't count towards evaluation
                continue
            elif pred_val < classes and gt_val < classes:
                confusion_matrix[gt_val, pred_val] += 1  # count the occurences
            else:
                print("Unknown predicted class detected", pred_val)
    intersection = np.diag(confusion_matrix)  # count the correct classifications i.e. the intersection
    union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - intersection  # sum cols(predicted) + sum rows(groundtruth) - intersection
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divisions by zero
        IOU = intersection/union
    meanIOU = np.nanmean(IOU)  # ignore the resulted NaNs
    print("confusion_matrix: %s" % confusion_matrix)
    print("IoU: %s" % IOU)
    print("mIoU: %f" % meanIOU)
    return confusion_matrix, IOU, meanIOU
