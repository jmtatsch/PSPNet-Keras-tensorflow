# A Keras implementation of Pyramid Scene Parsing Networks [PSPNet](https://github.com/hszhao/PSPNet)

Original Paper and implementation:

```
@inproceedings{zhao2017pspnet,
  author = {Hengshuang Zhao and
            Jianping Shi and
            Xiaojuan Qi and
            Xiaogang Wang and
            Jiaya Jia},
  title = {Pyramid Scene Parsing Network},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
```

# Installation:

```bash
python3 setup.py install
```

# Usage:

```bash
python3 pspnet.py
python3 pspnet.py -m pspnet101_cityscapes -i ../example_images/munster_000013_000019_leftImg8bit.png -o ../example_results/munster_000013_000019_leftImg8bit.png -s -f
python3 pspnet.py -m pspnet101_voc2012 -i ../example_images/000129.jpg -o ../example_results/000129.png -s -f
python3 pspnet.py -m pspnet50_ade20k -i ../example_images/ADE_val_00000435.jpg -o ../example_results/ADE_val_00000435.png -s -f
python3 pspnet.py -m pspnet50_ade20k -i ../example_images/ade20k.jpg -o ../example_results/ade20k.png -s -f

```

# Results:

## ADE20K:
<img src="example_images/ade20k.jpg" alt="Input" width="250"><img src="example_results/ade20k_seg.png" alt="Segmentation Keras" width="250"><img src="example_results/ade20k_seg_pycaffe.jpg" alt="Segmentation Original Pycaffe implementation" width="250"><img src="example_results/ade20k_seg_blended.png" width="250"><img src="example_results/ade20k_probs.png" alt="Uncertainty" width="250">

<img src="example_images/ADE_val_00000435.jpg" alt="Input" width="250"><img src="example_results/ADE_val_00000435_seg.png" alt="Segmentation Keras" width="250"><img src="example_groundtruth/ADE_val_00000435_seg.png" alt="Groundtruth" width="250"><img src="example_results/ADE_val_00000435_seg_blended.png" width="250"><img src="example_results/ADE_val_00000435_probs.png" alt="Uncertainty" width="250">

## CityScapes

<img src="example_images/munster_000013_000019_leftImg8bit.png" alt="Input" width="250"><img src="example_results/munster_000013_000019_leftImg8bit_seg.png" alt="Segmentation Keras" width="250"><img src="example_groundtruth/munster_000013_000019_gtFine_colors.png" alt="Groundtruth" width="250"><img src="example_results/munster_000013_000019_leftImg8bit_seg_blended.png" width="250"><img src="example_results/munster_000013_000019_leftImg8bit_probs.png" alt="Uncertainty" width="250">

## Pascal Voc 2012

<img src="example_images/000129.jpg" alt="Input" width="250"><img src="example_results/000129_seg.png" alt="Segmentation Keras" width="250"><img src="example_groundtruth/000129.png" alt="Groundtruth" width="250">
<img src="example_results/000129_seg_blended.png" width="250"><img src="example_results/000129_probs.png" alt="Uncertainty" width="250">

# Converting caffe weights

Converted trained weights are needed to run the network and will be downloaded from dropbox the first time you use a model. The weights of the original caffemodel were converted with weight_converter.py as follows:

```bash
python3 weight_converter.py <path to .prototxt> <path to .caffemodel>
```

Running the converter needs the compiled original PSPNet caffe code and pycaffe.
