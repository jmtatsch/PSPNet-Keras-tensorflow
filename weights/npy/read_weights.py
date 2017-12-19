from __future__ import print_function

import numpy as np
weights = np.load("pspnet101_cityscapes.npy").item()
settable_weights = 0
for layer, value in weights.items():
    print(layer)
    for attrib, vals in weights[layer].items():
        if attrib == "weights":
            print("weights: ", vals.shape)
        else:
            print(attrib)
    settable_weights += 1
print("Total settable weights %i" % settable_weights)
