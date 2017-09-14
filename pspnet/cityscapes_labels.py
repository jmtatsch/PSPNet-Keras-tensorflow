#!/usr/bin/python
"""The Cityscapes labels."""

from label import Label

labels = [Label('road', 0, (128, 64, 128)),
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

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
