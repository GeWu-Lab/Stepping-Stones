from itertools import chain
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import cv2
from transformers import Mask2FormerImageProcessor
import json
from PIL import Image

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def color_mask_to_label(mask_array, v_pallete):
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    color_mask_PIL = resize_img(224, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    return color_label 

def custom_collate(batch):
    mask_recs = [item['mask_recs'] for item in batch]
    image_size = [item['image_size'] for item in batch]
    vid = [item['vid'] for item in batch]
    feat_aud = [item['feat_aud'] for item in batch]
    feat_aud = torch.stack(feat_aud)
    pixel_values = [item['pixel_values'] for item in batch]
    pixel_values = torch.stack(pixel_values)
    pixel_mask = [item['pixel_mask'] for item in batch]
    pixel_mask = torch.stack(pixel_mask)
    class_labels = [item['class_labels'] for item in batch]
    mask_labels = [item['mask_labels'] for item in batch]
    res = {
        "mask_recs": mask_recs,
        "image_size": image_size,
        "vid": vid,
        "feat_aud": feat_aud,
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }
    return res

def get_train_dataloader(args):
    train_dataset_v1s = AVSBench(args,'train', "v1s")
    train_dataset_v1m = AVSBench(args,'train', "v1m")
    train_dataset_v2 = AVSBench(args,'train', "v2")
    train_loader_v1s = torch.utils.data.DataLoader(train_dataset_v1s, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, collate_fn=custom_collate)
    train_loader_v1m = torch.utils.data.DataLoader(train_dataset_v1m, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, collate_fn=custom_collate)
    train_loader_v2 = torch.utils.data.DataLoader(train_dataset_v2, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
                                     pin_memory=True, collate_fn=custom_collate)

    train_loader = chain(train_loader_v1s, train_loader_v1m, train_loader_v2)
    return train_loader

def get_val_dataloader(args,split):
    val_dataset_v1s = AVSBench(args,split, "v1s")
    val_dataset_v1m = AVSBench(args,split, "v1m")
    val_dataset_v2 = AVSBench(args,split, "v2")
    val_loader_v1s = torch.utils.data.DataLoader(val_dataset_v1s, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True, collate_fn=custom_collate)
    val_loader_v1m = torch.utils.data.DataLoader(val_dataset_v1m, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True, collate_fn=custom_collate)
    val_loader_v2 = torch.utils.data.DataLoader(val_dataset_v2, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=True, collate_fn=custom_collate)

    val_loader = chain(val_loader_v1s,val_loader_v1m, val_loader_v2)
    return val_loader

class AVSBench(Dataset):
    def __init__(self, args, split, ver):
        self.ver = ver
        self.data_base_path = args.data_path
        self.data_path = os.path.join(self.data_base_path,ver)
        self.split = split
        meta_path = f'{self.data_base_path}/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        sub_data = metadata[metadata['label'] == ver] 
        self.metadata = sub_data[sub_data['split'] == split] 

        self.frame_num = 10 if ver == 'v2' else 5  
        if self.ver == 'v1s' and self.split=='train':
            self.frame_num = 5

        self.pallete = get_v2_pallete(f'{self.data_base_path}/label2idx.json')
        self.feat_path=args.feature_path

        self.img_process = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",cache_dir="/data/users/juncheng_ma/.models_o/", local_files_only=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid = df_one_video['uid']
        # Audio data
        feat_aud_p = f'{self.feat_path}/{self.ver}_vggish_embs/{vid}.npy'
        feat_aud = torch.from_numpy(np.load(feat_aud_p)).squeeze().detach()[:self.frame_num]
        if feat_aud.shape[0] != 5 and self.ver == 'v1s':
            while feat_aud.shape[0] != 5:
                print(f'> warning: find {feat_aud.shape[0]}/5 audio clips in {vid}, repeat the last clip.')
                feat_aud = torch.concat([feat_aud, feat_aud[-1].view(1, -1)], dim=0)
        # Vision data
        image_list, color_label_list = [],[]
        for _idx in range(self.frame_num):  
            _idx=0 if self.split == 'train' and self.ver == 'v1s' else _idx
            path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image
            image = cv2.imread(path_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1)
            path_mask = f'{self.data_path}/{vid}/labels_rgb/{_idx}.png'
            color_label = load_color_mask_in_PIL_to_Tensor(path_mask, v_pallete=self.pallete, split=self.split).squeeze(0)
            color_label_list.append(color_label)
            image_list.append(image)
        # process image inputs into Mask2former form.
        image_inputs = self.img_process.preprocess(image_list, color_label_list,return_tensors="pt")
        image_inputs["mask_recs"] = torch.stack(color_label_list)
        # size when test/val
        image_inputs["image_size"] = [(224,224)]
        image_inputs["vid"] = vid
        image_inputs["feat_aud"] = feat_aud
        return image_inputs


