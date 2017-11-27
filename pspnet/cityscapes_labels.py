#!/usr/bin/python
"""The Cityscapes labels."""

from label import Label

train_labels = [Label('road', 0, (128, 64, 128)),
                Label('sidewalk', 1, (244, 35, 232)),
                Label('building', 2, (70, 70, 70)),
                Label('wall', 3, (102, 102, 156)),
                Label('fence', 4, (190, 153, 153)),
                Label('pole', 5, (153, 153, 153)),
                Label('traffic light', 6, (250, 170, 30)),
                Label('traffic sign', 7, (220, 220,  0)),
                Label('vegetation', 8, (107, 142, 35)),
                Label('terrain', 9, (152, 251, 152)),
                Label('sky', 10, (70, 130, 180)),
                Label('person', 11, (220, 20, 60)),
                Label('rider', 12, (255,  0,  0)),
                Label('car', 13, (0,  0, 142)),
                Label('truck', 14, (0,  0, 70)),
                Label('bus', 15, (0, 60, 100)),
                Label('train', 16, (0, 80, 100)),
                Label('motorcycle', 17, (0,  0, 230)),
                Label('bicycle', 18, (119, 11, 32))]

labels = [Label('unlabeled',  0, (0,  0,  0)),
          Label('ego vehicle',  1, (0,  0,  0)),
          # Label('rectification border',  2, (0,  0,  0)),
          # Label('out of roi',  3, (0,  0,  0)),
          # Label('static',  4,  (0,  0,  0)),
          Label('dynamic',  5, (111, 74,  0)),
          Label('ground',  6, (81,  0, 81)),
          Label('road',  7, (128, 64, 128)),
          Label('sidewalk',  8, (244, 35, 232)),
          Label('parking',  9, (250, 170, 160)),
          Label('rail track', 10,   (230, 150, 140)),
          Label('building', 11,    (70, 70, 70)),
          Label('wall', 12,    (102, 102, 156)),
          Label('fence', 13,      (190, 153, 153)),
          Label('guard rail', 14,      (180, 165, 180)),
          Label('bridge', 15,      (150, 100, 100)),
          Label('tunnel', 16,     (150, 120, 90)),
          Label('pole', 17,  (153, 153, 153)),
          # Label('polegroup', 18,      (153, 153, 153)),
          Label('traffic light', 19,        (250, 170, 30)),
          Label('traffic sign', 20,       (220, 220,  0)),
          Label('vegetation', 21,     (107, 142, 35)),
          Label('terrain', 22,       (152, 251, 152)),
          Label('sky', 23,     (70, 130, 180)),
          Label('person', 24,       (220, 20, 60)),
          Label('rider', 25,       (255,  0,  0)),
          Label('car', 26,      (0,  0, 142)),
          Label('truck', 27,       (0,  0, 70)),
          Label('bus', 28,         (0, 60, 100)),
          Label('caravan', 29,     (0,  0, 90)),
          Label('trailer', 30,      (0,  0, 110)),
          Label('train', 31,       (0, 80, 100)),
          Label('motorcycle', 32,      (0,  0, 230)),
          Label('bicycle', 33,       (119, 11, 32)),
          Label('license plate', -1, (0, 0, 142))]

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in train_labels}
full_id2label = {label.id: label for label in labels}
