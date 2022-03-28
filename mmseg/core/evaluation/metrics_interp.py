import torch
import mmcv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2


import pdb

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice



def compute_errors(pred, gt):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        return rmse


def relative_error(preds, depth):
        n_pixels = depth.shape[0] * depth.shape[1]
        d = preds - depth
        term_1 = np.power(d.reshape([n_pixels]), 2).mean()
        term_2 = np.power(d.reshape([n_pixels]).sum(), 2) / ((n_pixels**2))
        bloss = term_1 -  term_2
        return bloss


def gradient_error(prediction, target):
        total = 0
        prediction = prediction.squeeze()

        for scale in range(2,4):
            step = np.power(2,scale)
            pred = prediction[::step, ::step]
            tget = target[::step, ::step]
            n_pixels = tget.shape[1] * tget.shape[0]
            diff = pred - tget
            grad_x = np.abs(diff[:, 1:] - diff[:, :-1])
            grad_y = np.abs(diff[1:, :] - diff[:-1, :])
            image_loss = np.sum(grad_x) + np.sum(grad_y)
            total += np.mean(image_loss) / n_pixels

        return total

def calc_R_k(A,d):
    Rkd = torch.zeros([512,20]).cuda()
    for i,di in enumerate(range(5,100,5)):
        mask = (d>(di-5)) & (d<=di)
        if mask.sum() == 0:
            continue
        for k in range(512):
            Rkd[k,i] = (mask * A[k,...]).sum() / mask.sum()
    return Rkd

def calc_C_k(A,d):
    Rkd = torch.zeros([512,5]).cuda()
    for i,di in enumerate({2,5,6,9,10}):
        mask = (d==di)
        if mask.sum() == 0:
            continue
        for k in range(512):
            Rkd[k,i] = (mask * A[k,...]).sum() / mask.sum()
    return Rkd


def eval_metrics(results,
                 gt_seg_maps,
                 metrics=['mIoU'],
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mse']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    losses = 0.0
    nlosses = 0.0
    rlosses = 0.0
    nrlosses = 0.0
    glosses = 0.0
    nglosses = 0.0
    #CLS_root = '/home/xshadow/Track1-Truth/'
    #test_files = ['OMA_374_037_CLS.tif','OMA_315_037_CLS.tif','OMA_364_004_CLS.tif','OMA_355_001_CLS.tif','OMA_342_001_CLS.tif','OMA_383_027_CLS.tif','OMA_381_006_CLS.tif','OMA_329_001_CLS.tif','OMA_331_008_CLS.tif']

    rmses = 0.0
    nrmses = 0.0

    FRk =  torch.zeros([512,20]).cuda()
    #FCk =  torch.zeros([512,5]).cuda()
    Fsik =  torch.zeros([512]).cuda()
    #Fcsik =  torch.zeros([512]).cuda()

    for i in range(len(results)):
        out = results[i].squeeze()
        target = gt_seg_maps[i]
        np.save('road-all_height_result.npy',out)
        #pdb.set_trace()

        #cls_label = cv2.imread(CLS_root + test_files[i],2)
        #Ck = calc_C_k(out, cls_label)
        
        #out = np.clip(out, 0 , 400)
        #for k in range(20):
        #Rk = calc_R_k(out,target)   #512,20
        #sik = torch.zeros([512]).cuda()
        #csik = torch.zeros([512]).cuda()
        '''
        for k in range(512):
            item1 = Rk[k].max().abs()
            item2 = (Rk[k].sum()-Rk[k].max()).abs() / 19.
            sik[k] = (item1 - item2) / (item1 + item2)

            #citem1 = Ck[k].max().abs()
            #citem2 = (Ck[k].sum()-Ck[k].max()).abs() / 4.
            #csik[k] = (citem1 - citem2) / (citem1 + citem2)
        
        print(i)
        FRk += Rk
        #FCk += Ck
        Fsik += sik
        #Fcsik += csik
        '''
        
    return 0
