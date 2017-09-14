#!/usr/bin/python
"""A generic label definition."""

from collections import namedtuple

Label = namedtuple('Label', [

    'name',
    'id',
    'color'
])
