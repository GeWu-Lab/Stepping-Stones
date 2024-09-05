import torch
import torch.nn as nn
from utils.utility import get_loss
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

    def forward(self, batch_data):
        len_img = len(batch_data["mask_recs"])
        image_sizes = batch_data["image_size"]
        image_sizes = len_img * image_sizes
        img_input = {}
        audio_emb = batch_data["feat_aud"].cuda().view(1, -1, 128)
        audio_emb = self.audio_proj(audio_emb)
        img_input['prompt_features_projected'] = audio_emb
        img_input['pixel_values'] = batch_data['pixel_values'].squeeze().view(-1, 3, 384, 384).cuda()
        img_input['pixel_mask'] = batch_data['pixel_mask'].squeeze().view(-1, 384, 384).cuda()
        img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"]]
        img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"]]
        avs_outputs = self.model(**img_input)
        with torch.no_grad():
            logits = self.image_processor.post_process_binary_segmentation(avs_outputs, target_sizes=image_sizes)
        loss_frame = get_loss(avs_outputs.loss,batch_data["mask_recs"],batch_data["task"])
        return loss_frame, logits

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

    def forward(self, batch_data):
        len_img = len(batch_data["mask_recs"])
        image_sizes = batch_data["image_size"]
        image_sizes = len_img * image_sizes
        img_input = {}
        audio_emb = batch_data["feat_aud"].cuda().view(1, -1, 128)
        audio_emb = self.audio_proj(audio_emb)
        img_input['prompt_features_projected'] = audio_emb
        img_input['pixel_values'] = batch_data['pixel_values'].squeeze().view(-1, 3, 384, 384).cuda()
        img_input['pixel_mask'] = batch_data['pixel_mask'].squeeze().view(-1, 384, 384).cuda()
        img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"]]
        img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"]]
        avs_outputs = self.model(**img_input)
        with torch.no_grad():
            logits = self.image_processor.post_process_binary_segmentation(avs_outputs, target_sizes=image_sizes)
        loss_frame = avs_outputs.loss
        return [loss_frame], logits


