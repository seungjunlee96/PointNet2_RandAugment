# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import random

from .pointcloud_transforms import *



def augment_list():  #  operations and their ranges

    l = [
        (ScaleX, 0, 1),
        (ScaleY, 0, 1),
        (ScaleZ, 0, 1),
        (NonUniformScale, 0 , 1),
        (Resize , 0 , 1),

        (RotateX, 0, 2 * np.pi),
        (RotateY, 0, 2 * np.pi),
        (RotateZ, 0, 2 * np.pi),
        (RandomAxisRotation, 0, 2 * np.pi),
        (RotatePerturbation, 0, 10),

        (Jitter, 0 , 10),
        (TranslateX, 0, 1),
        (TranslateY, 0, 1),
        (TranslateZ, 0, 1),
        (NonUniformTranslate, 0 , 0.1),

        (RandomDropout, 0.5 , 0.875),
        (RandomErase, 0, 0.5),

        (ShearXY, 0 , 0.5 ),
        (ShearYZ, 0 , 0.5 ),
        (ShearXZ, 0 , 0.5 ),

        (GlobalAffine , 0 , 0.01),
        #(PiecewiseShear, 0 , 10),
        ]

    return l


class RandAugment:
    def __init__(self, n, m):
        """
        The number of augmentations = 20
        N : The number of augmentation choice
        M : magnitude of augmentation
        """
        self.n = n
        self.m = m      # [0, 10]
        self.augment_list = augment_list()

    def __call__(self, points):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            points = op(points, val)

        return points