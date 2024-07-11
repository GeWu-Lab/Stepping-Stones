from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import matplotlib.pyplot as plt


class AdaptiveAVS_AVSS(nn.Module):
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
        self.class_num=args.class_num
        self.prior_dir=args.mask_dir

    def forward(self, batch_data):
        len_img = len(batch_data["mask_recs"][0])
        image_sizes = batch_data["image_size"][0]
        image_sizes = len_img * image_sizes
        img_input = {}
        audio_emb = batch_data["feat_aud"][0].cuda().view(1, -1, 128)
        audio_emb = self.audio_proj(audio_emb)
        img_input['prompt_features_projected'] = audio_emb
        img_input['pixel_values'] = batch_data['pixel_values'][0].squeeze().view(-1, 3, 384, 384).cuda()
        img_input['pixel_mask'] = batch_data['pixel_mask'][0].squeeze().view(-1, 384, 384).cuda()
        img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"][0]]
        img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"][0]]
        img_input["first_stage_results"] = (batch_data["mask_recs"][0]!=0).float().cuda() if self.training else torch.load(f'{self.prior_dir}/{batch_data["vid"][0]}.pth').float()
        avss_outputs = self.model(**img_input)
        with torch.no_grad():
            pred_instance_map = self.image_processor.post_process_semantic_segmentation(
                self.class_num, avss_outputs, target_sizes=image_sizes)
        loss_frame = avss_outputs.loss
        return [loss_frame], pred_instance_map


