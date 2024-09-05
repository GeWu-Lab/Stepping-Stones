import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
import cv2
import os
from transformers import Mask2FormerImageProcessor
import json
from PIL import Image
from towhee import pipe, ops
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
    mask_recs = batch[0]['mask_recs']
    image_size = batch[0]['image_size']
    vid = batch[0]['vid']
    feat_aud = batch[0]['feat_aud']
    pixel_values = batch[0]['pixel_values'] 
    pixel_mask = batch[0]['pixel_mask']
    class_labels = batch[0]['class_labels']
    mask_labels = batch[0]['mask_labels']
    task=batch[0]['task']
    res = {
        "mask_recs": mask_recs,
        "image_size": image_size,
        "vid": vid,
        "feat_aud": feat_aud,
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
        "task":task
    }
    return res
def get_train_dataloader(args):
    if args.task=="avss":
        return get_avss_train_dataloader(args)
    else:
        return get_avs_train_dataloader(args)

def get_avss_train_dataloader(args):
    train_dataset_v1s = AVSBench(args,'train', "v1s")
    train_dataset_v1m = AVSBench(args,'train', "v1m")
    train_dataset_v2 = AVSBench(args,'train', "v2")
    train_loader_v1s = torch.utils.data.DataLoader(train_dataset_v1s, batch_size=args.bs, shuffle=True,num_workers=10,
                                      pin_memory=True, collate_fn=custom_collate)
    train_loader_v1m = torch.utils.data.DataLoader(train_dataset_v1m, batch_size=args.bs, shuffle=True, 
                                      pin_memory=True, collate_fn=custom_collate)
    train_loader_v2 = torch.utils.data.DataLoader(train_dataset_v2, batch_size=args.bs, shuffle=True, 
                                     pin_memory=True, collate_fn=custom_collate)

    train_loader = ConcatDataset([train_loader_v1s, train_loader_v1m, train_loader_v2])
    return train_loader
def get_avs_train_dataloader(args):
    train_dataset = AVSBench(args,'train', args.task)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=10,collate_fn=custom_collate)
    return train_loader
def get_val_dataloader(args,split):
    if args.task=="avss":
        return get_avss_val_dataloader(args,split)
    else:
        return get_avs_val_dataloader(args,split)
def get_avss_val_dataloader(args,split):
    val_dataset_v1s = AVSBench(args,split, "v1s")
    val_dataset_v1m = AVSBench(args,split, "v1m")
    val_dataset_v2 = AVSBench(args,split, "v2")
    val_loader_v1s = torch.utils.data.DataLoader(val_dataset_v1s, batch_size=1, shuffle=False,
                                      pin_memory=True, collate_fn=custom_collate)
    val_loader_v1m = torch.utils.data.DataLoader(val_dataset_v1m, batch_size=1, shuffle=False, 
                                      pin_memory=True, collate_fn=custom_collate)
    val_loader_v2 = torch.utils.data.DataLoader(val_dataset_v2, batch_size=1, shuffle=False,
                                     pin_memory=True, collate_fn=custom_collate)

    val_loader = ConcatDataset([val_loader_v1s,val_loader_v1m, val_loader_v2])
    return val_loader
def get_avs_val_dataloader(args,split):
    train_dataset = AVSBench(args,split, args.task)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=10,collate_fn=custom_collate)
    return train_loader


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
        self.audio_vggish_pipeline = (   # pipeline building
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.vggish())
                .output('vecs')
        )

        self.img_process = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",cache_dir=args.model_dir, local_files_only=True)

    def __len__(self):
        return len(self.metadata)
    
    def get_audio_emb(self, wav_path):
        """ wav string path. """ 
        emb = torch.tensor(self.audio_vggish_pipeline(wav_path).get()[0])
        return emb

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid = df_one_video['uid']
        # Audio data
        rec_audio = f'{self.data_path}/{vid}/audio.wav'
        feat_aud = self.get_audio_emb(rec_audio)
        if feat_aud.shape[0] != 5 and self.ver == 'v1s':
            while feat_aud.shape[0] != 5:
                feat_aud = torch.concat([feat_aud, feat_aud[-1].view(1, -1)], dim=0)
        # Vision data
        image_list, label_list = [],[]
        for _idx in range(self.frame_num):  
            _idx=0 if self.split == 'train' and self.ver == 'v1s' else _idx
            path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image
            image = cv2.imread(path_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1)
            path_mask = f'{self.data_path}/{vid}/labels_rgb/{_idx}.png'
            mask_cv2 = cv2.imread(path_mask)
            mask_cv2 = cv2.resize(mask_cv2, (224, 224))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            ground_truth_mask = torch.as_tensor((mask_cv2 > 0) , dtype=torch.float32)
            image_list.append(image)
            label_list.append(ground_truth_mask)
        # process image inputs into Mask2former form.
        image_inputs = self.img_process.preprocess(image_list, label_list,return_tensors="pt")
        image_inputs["mask_recs"] = torch.stack(label_list)
        # size when test/val
        image_inputs["image_size"] = [(224,224)]
        image_inputs["vid"] = vid
        image_inputs["feat_aud"] = feat_aud
        image_inputs["task"]=self.ver
        return image_inputs



