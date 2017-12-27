#!/usr/bin/python
"""Setup script for a Keras/Tensorflow implementation of Pyramid Scene Parsing Networks."""

from setuptools import setup


config = {
    'name': 'pspnet',
    'description': 'A Keras implementation of Pyramid Scene Parsing Networks',
    'author': 'Vlad Kryvoruchko, Chaoyue Wang, Jeffrey Hu & Julian Tatsch',
    'url': 'https://github.com/jmtatsch/PSPNet-Keras-tensorflow',
    'author_email': 'julian@tatsch.it',
    'version': '0.1',
    'dependency_links': ["https://github.com/jmtatsch/cityscapesScripts/tarball/master#egg=cityscapesscripts-0.1"],
    'install_requires': ['numpy', 'scipy', 'pillow', 'matplotlib', 'keras==2.1.2',
                         'h5py', 'requests', 'cityscapesscripts'],
    'packages': ['pspnet'],
    'package_data': {'pspnet': ['example_images/*.*', 'example_results/*.*']},
    'entry_points': {'console_scripts': ['pspnet = pspnet.pspnet:main']},
}

setup(**config)
