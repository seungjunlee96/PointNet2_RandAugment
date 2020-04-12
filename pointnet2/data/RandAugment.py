from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import random

import numpy as np
import torch
from pointnet2.data.data_utils import angle_axis

def RandomFlipX(points , v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:,0] *= -1
    return points

def RandomFlipY(points , v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:,1] *= -1
    return points

def RandomFlipZ(points , v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:,2] *= -1
    return points

def ScaleX(pts,v): # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 0 ] *= scaler
    return pts

def ScaleY(pts,v): # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 1 ] *= scaler
    return pts

def ScaleZ(pts,v): # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 2 ] *= scaler
    return pts

def Resize(pts,v):
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 0:3 ] *= scaler
    return pts

def NonUniformScale(pts,v): # Resize in [0.5 , 1.5]
    assert 0 <= v <= 0.5
    scaler = np.random.uniform( low = 1 - v, high =  1 + v, size = 3 )
    pts[:, 0:3] *= torch.from_numpy(scaler).float()
    return pts


def RotateX(points,v): # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([1. , 0. , 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points

def RotateY(points,v): # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0. , 1. , 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points


def RotateZ(points,v): # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0. , 0. , 1.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points

def RandomAxisRotation(points,v):
    assert 0 <= v <= 2 * np.pi
    axis = np.random.randn(3)
    axis /= np.sqrt((axis**2).sum())

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points

def RotatePerturbation(points,v):
    assert 0 <= v <= 10
    v = int(v)
    angle_sigma = 0.1 * v
    angle_clip = 0.1 * v
    n_idx = 50 * v

    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
    Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
    Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

    rotation_matrix = Rz @ Ry @ Rx

    center = torch.mean(points[:,0:3], dim = 0)
    idx = np.random.choice(points.size(0),n_idx)

    perturbation = points[idx, 0:3] - center
    points[idx, :3] += (perturbation @ rotation_matrix.t()) - perturbation

    normals = points.size(1) > 3
    if normals:
        pc_normals = points[idx, 3:]
        points[idx, 3:] = pc_normals @ rotation_matrix.t()

    return points


def Jitter(points,v):
    assert 0.0 <= v <= 10
    v = int(v)

    sigma = 0.1 * v
    n_idx = 50 * v

    idx = np.random.choice(points.size(0),n_idx)
    jitter = sigma * (np.random.random([n_idx, 3]) - 0.5)
    points[idx, 0:3] += torch.from_numpy(jitter).float()
    return points

def PointToNoise(points,v):
    assert 0 <= v <= 0.5
    mask = np.random.random(points.size(0)) < v
    noise_idx = [idx for idx in range(len(mask)) if mask[idx] == True]
    pts_rand = 2 * (np.random.random([len(noise_idx), 3]) - 0.5) + np.mean(points[:,0:3].numpy(), axis= 0)

    points[noise_idx, 0:3] = torch.from_numpy(pts_rand).float()

    return points

def UniformTranslate(points ,v):
    assert 0 <= v <= 1
    translation = (2 * np.random.random() - 1 ) * v
    points[:, 0:3] += translation
    return points

def NonUniformTranslate(points ,v):
    assert 0 <= v <= 1
    translation = (2 * np.random.random(3) - 1 ) * v
    points[:, 0:3] +=  torch.from_numpy(translation).float()
    return points

def RandomDropout(points,v):
    assert 0.3 <= v <= 0.875
    dropout_rate = v
    drop = torch.rand(points.size(0)) < dropout_rate
    save_idx = np.random.randint(points.size(0))
    points[drop] = points[save_idx]
    return points

def RandomErase(points,v):
    assert 0 <= v <= 0.5
    "v : the radius of erase ball"
    random_idx = np.random.randint(points.size(0))
    mask = torch.sum((points[:,0:3] - points[random_idx,0:3]).pow(2), dim = 1) < v ** 2
    points[mask] = points[random_idx]
    return points
#
# def DBSCAN(points,v):
#     "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
#     assert 0 <= v <= 10
#     from sklearn.cluster import DBSCAN
#     eps = 0.03 * v
#     min_samples =  v
#     clustering = DBSCAN(eps = eps , min_samples = min_samples).fit(points[:,0:3].numpy())
#     for label in set(clustering.labels_):
#         mask = (clustering.labels_ == label)
#         points[mask,0:3] = torch.mean(points[mask,0:3] , dim = 0)
#
#     return points

def ShearXY(points,v):
    assert 0 <= v <= 0.5
    a , b  = v * (2 * np.random.random(2) - 1)
    shear_matrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [a, b, 1]])
    shear_matrix = torch.from_numpy(shear_matrix).float()

    points[:,0:3] = points[:, 0:3] @ shear_matrix.t()
    return points

def ShearYZ(points,v):
    assert 0 <= v <= 0.5
    b , c  = v * (2 * np.random.random(2) - 1)
    shear_matrix = np.array([[1, b, c],
                             [0, 1, 0],
                             [0, 0, 1]])
    shear_matrix = torch.from_numpy(shear_matrix).float()

    points[:,0:3] = points[:, 0:3] @ shear_matrix.t()
    return points

def ShearXZ(points,v):
    assert 0 <= v <= 0.5
    a , c  = v * (2 * np.random.random(2) - 1)
    shear_matrix = np.array([[1, 0, 0],
                             [a, 1, c],
                             [0, 0, 1]])
    shear_matrix = torch.from_numpy(shear_matrix).float()
    points[:,0:3] = points[:, 0:3] @ shear_matrix.t()

    return points


def GlobalAffine(points,v):
    assert 0 <= v <= 1

    affine_matrix = torch.from_numpy(np.eye(3) + np.random.randn(3,3) * v).float()
    points[:,0:3] = points[:, 0:3] @ affine_matrix.t()

    return points



# def MixSample(points,v):
    # assert 0 <= v <= 1
    # dropout_rate = v
    # drop = torch.rand(points.size(0)) < dropout_rate
    # translation = (2 * np.random.random(3) - 1) * v
    # points[drop,0:3] += torch.from_numpy(translation).float()
    # return points

# def GridDistortion(points,v):
#     pass

# def ElasticDeformation(points,v):
#     pass
#
# def Voxelize(points,v):
#     assert 0 <= v <= 5
#     radius = 1.
#     vsize = int(100/v)
#     voxel = 2 * radius / vsize
#     locations = ((points[:,0:3] + radius) / voxel).int()x
#     for loc in set(locations):
#         mask = (locations == loc)
#         points[mask,0:3] = loc * voxel - radius
#
#     return points


# def PiecewiseShear(points,v):
#     assert 0 <= v <= 10
#
#     radius = 0.01 * int(v)
#     n_piece = 5 * int(v)
#     magnitude = 0.05 * v
#
#     for _ in range(n_piece):
#         pts = points[:,0:3]
#         centroid_pt = pts[np.random.randint(points.size(0))]
#         box_idx = ((torch.abs(pts - centroid_pt) < radius ).sum(axis=1) == 3 )
#
#         shear_lst = [ShearXY , ShearXZ , ShearYZ]
#         shear_choice = random.choices(shear_lst, k = 1)
#         for op in shear_choice:
#             points[box_idx,0:3] = op(pts[box_idx], magnitude)
#
#     return points

def Identity(points , v):
    return points


def augment_list():  #  operations and their ranges

    l = (
        (Identity , 0 ,10),

        (RandomFlipX, 0, 1),
        (RandomFlipY, 0, 1),
        (RandomFlipZ, 0, 1),

        (ScaleX, 0, 0.5),
        (ScaleY, 0, 0.5),
        (ScaleZ, 0, 0.5),
        (NonUniformScale, 0 , 0.5),
        (Resize , 0 , 0.5),

        (RotateX, 0, 2 * np.pi),
        (RotateY, 0, 2 * np.pi),
        (RotateZ, 0, 2 * np.pi),
        (RandomAxisRotation, 0, 2 * np.pi),

        (RotatePerturbation, 0, 10),
        (Jitter, 0 , 10),

        (UniformTranslate, 0 , 0.5),
        (NonUniformTranslate, 0 , 0.5),

        (RandomDropout, 0.3 , 0.875),
        (RandomErase, 0, 0.5),
        (PointToNoise, 0 , 0.5),

        (ShearXY, 0 , 0.5 ),
        (ShearYZ, 0 , 0.5 ),
        (ShearXZ, 0 , 0.5 ),
        (GlobalAffine , 0 , 0.15),
    )

    return l

def UpdateNM(N,M,increase = True):
    N_tmp, M_tmp = N, M
    if increase:
        if np.random.random() > 0.5:
            N += 1
        elif M < 10:
            M += 1
        print("\n Increase N,M from ({},{}) to ({} ,{}) \n".format(N_tmp, M_tmp, N, M))
    elif increase == False:
        if np.random.random() > 0.5 and N > 1:
            N -= 1
        elif M > 1:
            M -= 1
        print("\n Decrease N,M from ({},{}) to ({} ,{}) \n".format(N_tmp, M_tmp, N, M))
    return N,M



class RandAugment:
    def __init__(self, n, m):
        """
        The number of augmentations = ?
        N : The number of augmentation choice
        M : magnitude of augmentation
        """
        self.n = n
        self.m = m      # [0, 10]
        self.augment_list = augment_list()
        self.epoch = 0

    def __call__(self, pc):
        assert 0 <= self.m <= 10

        if pc.dim() == 3:
            bsize = pc.size()[0]
            for i in range(bsize):
                points = pc[i,:,:]
                ops = random.choices(self.augment_list, k=self.n)
                for op, minval, maxval in ops:
                    val = (float(self.m) / 10) * float(maxval - minval) + minval
                    points = op(points, val)
        elif pc.dim() == 2:
            points = pc
            ops = random.choices(self.augment_list, k=self.n)
            for op, minval, maxval in ops:
                val = (float(self.m) / 10) * float(maxval - minval) + minval
                points = op(points, val)
        return pc

    def UpdateNM(self,increase=True):
        N_tmp, M_tmp = self.N, self.M
        if increase:
            if np.random.random() > 0.5:
                self.N += 1
            elif self.M < 10:
                self.M += 1
            print("\n Increase N,M from ({},{}) to ({} ,{}) \n".format(N_tmp, M_tmp, self.N, self.M))
        elif increase == False:
            if np.random.random() > 0.5 and self.N > 1:
                self.N -= 1
            elif self.M > 1:
                self.M -= 1
            print("\n Decrease N,M from ({},{}) to ({} ,{}) \n".format(N_tmp, M_tmp, self.N, self.M))

