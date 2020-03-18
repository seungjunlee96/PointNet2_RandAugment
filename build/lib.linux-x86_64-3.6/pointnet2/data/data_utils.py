from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import numpy as np

def FarthestPointSample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    References : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class PointCloudNormalize(object):
    def __call__(self, points):
        """
        points : numpy array
        """
        centroid = np.mean(points, axis=0)
        points -= centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points /= m
        return points

class PointCloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

class RandomFlipX(object):
    def __init__(self, p = 0.5 ):
        self.p = p

    def __call__(self, points):
        if np.random.random() < self.p:
            points[:,0] *= -1
        return points

class RandomFlipY(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, points):
        if np.random.random() < self.p:
            points[:, 1] *= -1
        return points

class RandomFlipZ(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, points):
        if np.random.random() < self.p:
            points[:, 2] *= -1
        return points