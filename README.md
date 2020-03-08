# RandAugment3D_PointNet2

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) with aggressive data augmentations and RandAugment for point cloud data to handle with 3D classification task

Unofficial PyTorch modification of RandAugment by [ildoonet's RandAugment](https://github.com/ildoonet/pytorch-randaugment)


* Supports Multi-GPU via `nn.DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_.

* Supports PyTorch version >= 1.0.0.  Use `v1.0 <https://github.com/erikwijmans/Pointnet2_PyTorch/releases/tag/v1.0>`_
  for support of older versions of PyTorch.

Setup
-----

* Install ``python`` -- This repo is tested with ``2.7``, ``3.5``, and ``3.6``


* Install dependencies

  ::

    pip install -r requirements.txt


* Building `_ext` module

  ::

    python setup.py build_ext --inplace


* Optionally, you can also install this repo as a package

  ::

    pip install -e .


## Reference By
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
[erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)<br>
[ildoonet's RandAugment](https://github.com/ildoonet/pytorch-randaugment)<br>
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
