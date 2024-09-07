import torch
import os
from config import args
from dataset import get_avss_val_dataloader
from avs_model import AdaptiveAVS
from scripts import test


if __name__ == '__main__':
    test_loader = get_avss_val_dataloader(args,"test")
    # Model
    model = AdaptiveAVS(args).cuda()
    model.load_state_dict(torch.load(args.ckpt_dir),strict=False)
    model.eval()
    res = test(model, test_loader, args)
    print(res)
