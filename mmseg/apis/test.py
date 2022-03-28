import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import pdb

import cv2, os, sys, matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/home/xshadow/LAM_Demo')

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad, attr_id, attr_gabor_generator
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient_GTA
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath, LinearPath, CosinePath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

import pdb


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_interp(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    window_size = 8  # Define windoes_size of D

    #w = 403  # The x coordinate of your select patch, 125 as an example
    #h = 161  # The y coordinate of your select patch, 160 as an example
         # And check the red box
         # Is your selected patch this one? If not, adjust the `w` and `h`.
    
    #w, h = 1603,497
    w, h = 297,630
    w,h = 181,128
    w,h = 1193,377
    w,h = 463,611
    w,h = 301,617
    w,h = 645,699
    w,h = 929,112
    w,h = 161,531

    w,h = 780,163
    w,h = 247,122
    w,h = 192,379
    w,h = 743,554
    w,h = 560,498
    w,h = 597,664
    w,h = 51,894
    #w,h = 387,609
    w,h = 237,967
    w,h = 694,879
    w,h = 743,554
    w,h = 597,664


    fold = 100; alpha = 0.4

    attr_objective = attribution_objective(attr_id, h, w, window=window_size)
    
    gaus_blur_path_func = LinearPath(fold)
    #gaus_blur_path_func = CosinePath(fold)

    for i, data in enumerate(data_loader):
        #with torch.no_grad():
            #result = model(return_loss=False, **data)
            #result = model(return_loss=False, img=[data['img']], img_metas=[data['img_metas']])
        interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient_GTA(data, model, attr_objective, gaus_blur_path_func, cuda=True)
        img_lr = data['img'][0].cpu().numpy().squeeze()
        img_lr = cv2.imread('/home/xshadow/GTA_height/images/validation_t/img--1036_-1540_299_15_278_0_0_k0.jpg')
        img_lr = cv2.imread('/home/xshadow/height_data/images/validation_syt/OMA_329_001_RGB.tif')
        #img_lr = cv2.resize(img_lr, (960,540))
        #img_lr = np.moveaxis(img_lr,0,2)
        position_pil = cv2_to_pil(img_lr)
        
        grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
        abs_normed_grad_numpy = grad_abs_norm(grad_numpy)

        saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=2)
        
        #saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
        #pdb.set_trace()
        blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + img_lr * alpha)
        #blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
        
        #'''
        pil = make_pil_grid(
            [position_pil,
            saliency_image_abs,
            blend_abs_and_input]
        )
        #'''
        pil.save('pppil.png')
        saliency_image_abs.save('pig_test.png')

        pdb.set_trace()

        simage = result.detach().cpu().numpy().squeeze()
        simage = (simage - simage.min()) / (simage.max() - simage.min()) * 255
        simage = simage.astype(np.uint8)
        
        simage = cv2.applyColorMap(simage, cv2.COLORMAP_JET)
        cv2.imwrite('tt.png', simage)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        #batch_size = data['img'][0].size(0)
        batch_size=1
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            #result = model(return_loss=False, **data)
            result = model(return_loss=False, img=[data['img']], img_metas=[data['img_metas']])

            #pdb.set_trace()

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        #batch_size = data['img'][0].size(0)
        batch_size=1
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.float32,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.float32, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.float32, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.float32, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
