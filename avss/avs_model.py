import torch
import torch.nn as nn
import cv2,os
import numpy as np
from transformers import Mask2FormerImageProcessor,Mask2FormerForUniversalSegmentation

class AdaptiveAVS(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.image_processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                                         cache_dir=args.model_dir,
                                                                         local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                                        cache_dir=args.model_dir,
                                                                        local_files_only=True,
                                                                        ignore_mismatched_sizes=True).cuda()
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.mask_path=args.mask_path

    def forward(self, batch_data):
        image_sizes = batch_data["image_size"][0]
        img_input = {}
        audio_emb = batch_data["feat_aud"].cuda().view(1, -1, 128)
        audio_emb = self.audio_proj(audio_emb)
        img_input['prompt_features_projected'] = audio_emb
        img_input['pixel_values'] = batch_data['pixel_values'].squeeze().view(-1, 3, 384, 384).cuda()
        img_input['pixel_mask'] = batch_data['pixel_mask'].squeeze().view(-1, 384, 384).cuda()
        img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"]]
        img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"]]
        img_input["audio_boost_mask"]=(batch_data["mask_recs"]!=0).float().cuda() if self.training else self.load_mask(batch_data["vid"],batch_data["subsets"]).cuda()
        avss_outputs = self.model(**img_input)
        with torch.no_grad():
            pred_mask = self.image_processor.post_process_semantic_segmentation(avss_outputs, target_sizes=image_sizes)
        return avss_outputs.loss, pred_mask
    def load_mask(self,vid,ver):
        paths=os.listdir(os.path.join(self.mask_path,ver,vid))
        masks=[]
        for i in range(len(paths)):
            masks.append((cv2.imread(os.path.join(self.mask_path,ver,vid,f"{i}.png"), cv2.IMREAD_GRAYSCALE)/255).astype(np.float32))
        masks=torch.tensor(np.stack(masks))
        return masks
    