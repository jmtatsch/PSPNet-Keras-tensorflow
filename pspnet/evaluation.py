#!/usr/bin/python
"""evaluate iou result with cityscapes tool."""
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists, args


def evaluate(pred_imgs, gt_imgs):
    """Thin wrapper around the cityscapes evaluation script."""
    print("Evaluating %s against %s" % (pred_imgs, gt_imgs))
    evaluateImgLists(pred_imgs, gt_imgs, args)
