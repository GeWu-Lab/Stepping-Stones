import os
import time
import torch
from config import args
import argparse
from favss_final_test.dataset import get_train_dataloader,get_val_dataloader
from models import AdaptiveAVS_AVSS
import copy
from scripts import test
import wandb
import random
import numpy as np

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = "offline"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_params(model):
    params_names = [
        'queries_embedder',
        'swin_adapt',
        'avs_adapt',
        'class_predictor',
        'prior_embed',
        "audio_proj"
    ]
    for name, param in model.named_parameters():
        param.requires_grad = False
        for _n in params_names:
                if _n in name: 
                    param.requires_grad = True  
    return params        
       
if __name__ == '__main__':
    # Fix seed
    set_seed(123)

    # dir to save checkpoint
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime('_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir
    wandb.init(
        project="AVSS task",
        config=args,
        name="Adaptive_AVS"
    )
    # Data
    train_loader=get_train_dataloader(args)
    val_loader = get_val_dataloader(args,"val")
    # Model
    model = AdaptiveAVS_AVSS().cuda()
    # Optimizer
    params=set_params(model)
    optimizer = torch.optim.AdamW(params,lr=args.lr, weight_decay=args.weight_dec, eps=1e-8, betas=(0.9, 0.999))

    train_losses = []
    miou, f_score = [], []
    max_miou=0
    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        train_loader_epoch = copy.deepcopy(train_loader)
        val_loader_epoch = copy.deepcopy(val_loader)
        
        model.train()
        losses = []
        for batch_idx, batch_data in enumerate(train_loader_epoch):
            loss_vid, _= model(batch_data)
            loss_vid = torch.mean(torch.stack(loss_vid))
            optimizer.zero_grad()
            loss_vid.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.01 ,norm_type=2)
            optimizer.step()
            print(f'loss_{idx_ep}_{batch_idx}: {loss_vid.item()}', end='\r')
            losses.append(loss_vid.item())
        loss = {"total_loss":np.mean(losses),}
        wandb.log(loss)

        model.eval()
        res = test(model, val_loader_epoch, args)
        miou.append(res["miou"])
        f_score.append(res["fscore"])
        if res["miou"] > max_miou:
            model_save_path = os.path.join(log_dir,"val_miou_best.pth")
            max_miou = res["miou"]
            torch.save(model, model_save_path)   
        wandb.log(res)

