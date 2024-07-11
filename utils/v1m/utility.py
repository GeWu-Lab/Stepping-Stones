import torch
from torch.nn import functional as F

import os
import cv2
import numpy as np
from PIL import Image

import pandas as pd
import pdb
from torchvision import transforms


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param:
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # return mask_iou_224(pred, target, eps=1e-7)
    # print(pred.shape, target.shape)
    assert len(pred.shape) == 3 and pred.shape == target.shape

    # print(torch.max(pred), torch.min(pred))
    # print(torch.max(target), torch.min(target))
    # input('max-min')
    pred = torch.sigmoid(pred)

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    #temp_pred = torch.sigmoid(pred)
    #pred = (temp_pred > 0.4).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, device='cuda:0'):
    if device.startswith('cuda'):
        prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
        thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path, pr_num=255, device='cuda:0'):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    pred = torch.sigmoid(pred)  # =======================================[important]
    # print(pred)
    # input()
    N = pred.size(0)
    beta2 = 0.1
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    # fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num, device=device)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    # fLog.close()

    return score.max().item()


def save_mask(pred_masks, save_base_path, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()

    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx]  # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png" % (video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')


def save_raw_img_mask(anno_file_path, raw_img_base_path, mask_base_path, split='test', r=0.5):
    df = pd.read_csv(anno_file_path, sep=',')
    df_test = df[df['split'] == split]
    count = 0
    for video_id in range(len(df_test)):
        video_name = df_test.iloc[video_id][0]
        raw_img_path = os.path.join(raw_img_base_path, video_name)
        for img_id in range(5):
            img_name = "%s.mp4_%d.png" % (video_name, img_id + 1)
            raw_img = cv2.imread(os.path.join(raw_img_path, img_name))
            mask = cv2.imread(os.path.join(mask_base_path, 'pred_masks', video_name, "%s_%d.png" % (video_name, img_id)))
            # pdb.set_trace()
            raw_img_mask = cv2.addWeighted(raw_img, 1, mask, r, 0)
            save_img_path = os.path.join(mask_base_path, 'img_add_masks', video_name)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_img_path, img_name), raw_img_mask)
        count += 1
    print(f'count: {count} videos')

