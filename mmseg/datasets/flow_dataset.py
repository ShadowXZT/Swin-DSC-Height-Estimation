import os
import numpy as np
import sys

from glob import glob
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from tqdm import tqdm

import json
import cv2

from .builder import DATASETS
from segmentation_models_pytorch.utils.meter import AverageValueMeter

from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from .pipelines import (
    UNITS_PER_METER_CONVERSION_FACTORS,
    convert_and_compress_prediction_dir,
    load_image,
    load_vflow,
    get_rms,
    get_r2_info,
    get_angle_error,
    get_r2,
    save_image,
)
from .pipelines import augment_vflow
from collections import defaultdict

RNG = np.random.RandomState(4321)

from mmcv.parallel import DataContainer as DC


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')



class args():
    augmentation = True
    downsample = 2
    #dataset_dir = "/home/xshadow/xdata/data/comp-train-half-res"
    dataset_dir = "/home/xshadow/xdata/data/test_rgb"
    test_sub_dir = 'test'
    backbone = 'resnet34'
    sample_size = None
    rgb_suffix = "j2k"
    nan_placeholder = 65535
    unit = 'm'


class Collect(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'



class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            #img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(to_tensor(results['gt_semantic_seg'][None,...]),stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@DATASETS.register_module()
class FlowDataset(BaseDataset):
    def __init__(
        self,
        sub_dir,
        args=args,
        rng=RNG,
    ):

        self.is_test = sub_dir == args.test_sub_dir
        self.rng = rng
        self.format_bundle = DefaultFormatBundle()
        self.collect_train = Collect(keys=['img','gt_semantic_seg','scale','xydir','agl'])
        #self.collect_train = Collect(keys=['img','gt_semantic_seg'])
        self.collect_test = Collect(keys=['img'])

        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(args.dataset_dir)
        rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))
        if rgb_paths == []: rgb_paths = list(dataset_dir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names
        agl_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif")
            for pth in rgb_paths
        )
        vflow_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_VFLOW")).with_suffix(".json")
            for pth in rgb_paths
        )

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            if args.sample_size is not None:
                self.paths_list = self.paths_list[: args.sample_size]
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.backbone, "imagenet"
        )

        self.args = args
        self.sub_dir = sub_dir

    def __getitem__(self, i):

        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
        else:
            rgb_path, vflow_path, agl_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
            agl = load_image(agl_path, self.args)
            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
            scale = vflow_data["scale"]
            if self.args.augmentation:
                image, mag, xdir, ydir, agl, scale = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        if self.is_test and self.args.downsample > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[0] / self.args.downsample),
                    int(image.shape[1] / self.args.downsample),
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))
        results = {}

        if self.is_test:
            results['img'] = image
            results['filename'] = str(rgb_path)
            results['ori_filename'] = str(rgb_path)
            results['img_shape']=(1024,1024,3)
            results['ori_shape']=(2048,2048,3)
            results['img_norm_cfg']=None
            results['flip_direction']=0
            results['pad_shape']=0
            results['scale_factor']=1.0
            results['flip']=False
            results = self.format_bundle(results)
            results = self.collect_test(results)
            return results
        else:
            results['img'] = image
            results['filename'] = str(rgb_path)
            results['ori_filename'] = str(rgb_path)
            results['xydir'] = xydir
            results['agl'] = agl
            #result['mag'] = mag
            results['img_shape']=2048
            results['ori_shape']=2048
            results['img_norm_cfg']=None
            results['flip_direction']=0
            results['pad_shape']=0
            results['scale_factor']=1.0
            results['flip']=False
            results['scale'] = scale
            results['gt_semantic_seg'] = mag
            results = self.format_bundle(results)
            results = self.collect_train(results)
            #results['seg_fields'].append('gt_semantic_seg')

            return results
            #return image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.paths_list)


