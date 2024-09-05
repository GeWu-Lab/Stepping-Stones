import torch
from utils import utility,pyutils
import cv2,os
from scripts.save_mask import save_batch_mask
N_CLASSES = 2
avg_meter_miou = pyutils.AverageMeter('miou')
avg_meter_F = pyutils.AverageMeter('F_score')
def test_avs(model, test_loader,  args):
    if args.save_mask:
        os.makedirs("save_masks", exist_ok=True)
        save_base_path=os.path.join("save_masks", args.task)
        os.makedirs(save_base_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            _, logits = model(batch_data)
            # print(vid_preds)
            logits = torch.stack(logits, dim=0).squeeze().cpu()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = batch_data["mask_recs"].squeeze()

            miou = utility.mask_iou(logits, vid_masks_t)
            F_score = utility.Eval_Fmeasure(logits, vid_masks_t, './logger', device=args.device)
            if args.save_mask:
                os.makedirs(os.path.join(save_base_path,batch_data["vid"]), exist_ok=True)
                save_batch_mask(logits,os.path.join(save_base_path,batch_data["vid"]))
            
            avg_meter_miou.add({'miou': miou.item()})
            avg_meter_F.add({'F_score': F_score})

    miou = round(avg_meter_miou.pop('miou'),6)
    f_score = round(avg_meter_F.pop('F_score'),6)
    
    res = {
        'miou': miou,
        'fscore': f_score,
    }

    return res
