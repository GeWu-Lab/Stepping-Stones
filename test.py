import torch
import os
from config import args
from favss_final_test.dataset import get_val_dataloader
from models import AdaptiveAVS_AVSS
from scripts import test


if __name__ == '__main__':
    test_loader = get_val_dataloader(args,"test")
    if args.save_mask:
        save_path=os.path.join(args.ckpt_dir,"save_masks")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    # Model
    model = AdaptiveAVS_AVSS().cuda()
    model.load_state_dict(torch.load(args.ckpt_dir+"/val_miou_best.pth").state_dict(),strict=False)
    model.eval()
    res = test(model, test_loader, args)
    print(res)
