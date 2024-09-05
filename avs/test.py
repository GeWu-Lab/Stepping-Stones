import torch
import os
from config import args
from dataset import get_val_dataloader
from avs_model import AdaptiveAVS_AVSS
from scripts import test_avs


if __name__ == '__main__':
    test_loader = get_val_dataloader(args,"test")
    if args.save_mask:
        os.makedirs("save_masks", exist_ok=True)
        save_path=os.path.join("save_masks", args.task)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    # Model
    model = AdaptiveAVS_AVSS(args).cuda()
    model.load_state_dict(torch.load(args.ckpt_dir),strict=False)
    model.eval()
    res = test_avs(model, test_loader, args)
    print(res)
