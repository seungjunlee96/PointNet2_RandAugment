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
        points : torch tensor

        returns points in range [-1,1]
        """
        centroid = torch.mean(points[:,0:3], axis=0)
        points[:,0:3] -= centroid
        m = torch.max(points[:,0:3].pow(2).sum(axis = 1).pow(0.5))
        points[:,0:3] /= m
        return points

class PointCloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

