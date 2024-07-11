import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
from torchvision import transforms

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import pdb

def miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    # predict = torch.argmax(output, 1) + 1
    # target = target.float() + 1
    predict = output+1
    target = target.float()+1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param:
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T)
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T)
    return miou, fscore, cls_count, vid_miou_list
