import os
import time
import torch
from config import args
from dataset import get_avss_train_dataloader,get_avss_val_dataloader
from avs_model import AdaptiveAVS
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
def get_fineture_model(args,log_dir):
    model = AdaptiveAVS(args).cuda()
    params_dict=torch.load(os.path.join(log_dir,"f_miou_best.pth"))
    new_dict={}
    for i in params_dict.keys():
        if not 'swin_adapt' in i:
            new_dict[i]=params_dict[i]
    model.load_state_dict(new_dict,strict=False)
    return model
def set_params(model, change):
    params_names = [
        'queries_embedder',
        'avs_adapt',
        'class_predictor',
        "audio_proj",
        "audio_embed",
        "silence_embed",
        "uncertain_embed"
    ]
    params=[]
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'swin_adapt' in name:
            if change:
                param.requires_grad = True  
                params.append(param)
            else:
                torch.nn.init.zeros_(param)
        for _n in params_names:
                if _n in name: 
                    param.requires_grad = True  
                    params.append(param)
    return params    


if __name__ == '__main__':
    # Fix seed
    #rank=init_dist()
    set_seed(3407)
    # dir to save checkpoint
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    #if rank==0:
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime('_%Y%m%d-%H%M%S')))
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(
        project="AVSS task",
        config=args,
        name="Adaptive_AVS"
    )
    wandb.require("core")
    
    # Data
    train_loader=get_avss_train_dataloader(args)
    val_loader = get_avss_val_dataloader(args,"val")
    # Model
    model = AdaptiveAVS(args).cuda()
    # Optimizer
    params=set_params(model, False)
    optimizer = torch.optim.AdamW(params,lr=args.lr, weight_decay=args.weight_dec, eps=1e-8, betas=(0.9, 0.999))
    train_losses = []
    miou = []
    max_miou=0

    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        if idx_ep==int(args.epochs//2):
            model=get_fineture_model(args,log_dir)
            params=set_params(model, True)
            optimizer = torch.optim.AdamW(params,lr=1e-4, weight_decay=args.weight_dec, eps=1e-8, betas=(0.9, 0.999))
        model.train()
        losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            loss_vid, _= model(batch_data)
            optimizer.zero_grad()
            loss_vid.backward()
            print(f'loss_{idx_ep}_{batch_idx}: {loss_vid.item()}', end='\r')
            #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.01 ,norm_type=2)
            optimizer.step()
            losses.append(loss_vid.item())
        loss = {"total_loss":np.mean(losses),}
        
        model.eval()
        res = test(model, val_loader, args)
        miou.append(res["miou"])
        if res["miou"] > max_miou:
            model_save_path = os.path.join(log_dir,"f_miou_best.pth")
            max_miou = res["miou"]
            torch.save(model.state_dict(), model_save_path)   
        wandb.log(res)
        print(miou)

