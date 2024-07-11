import torch
from utils import miou_fscore
from scripts.save_mask import save_batch_raw_mask

def test(model, test_loader,  args):
    N_CLASSES = 71
    model.eval()
    miou_pc= torch.zeros((N_CLASSES))  # miou value per class (total sum)
    Fs_pc= torch.zeros((N_CLASSES))  # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES))  # count per class
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            _, vid_preds = model(batch_data)
            vid_preds_t = torch.stack(vid_preds[0], dim=0).squeeze().cuda()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = torch.stack(batch_data["mask_recs"], dim=0).squeeze().cuda().float()
            if args.save_mask:
                save_batch_raw_mask(batch_idx,vid_preds,args)
            _miou_pc, _fscore_pc, _cls_pc, _ = miou_fscore(vid_preds_t, vid_masks_t, N_CLASSES, len(vid_masks_t))
            miou_pc += _miou_pc
            cls_pc += _cls_pc
            Fs_pc += _fscore_pc

    miou_pc = miou_pc / cls_pc
    miou_pc[torch.isnan(miou_pc)] = 0
    miou = torch.mean(miou_pc).item()

    miou_noBg = torch.mean(miou_pc[1:]).item()
    f_score_pc = Fs_pc / cls_pc
    f_score_pc[torch.isnan(f_score_pc)] = 0
    f_score = torch.mean(f_score_pc).item()
    f_score_noBg = torch.mean(f_score_pc[1:]).item()

    res = {
        'miou': miou,
        'miou_nb': miou_noBg,
        'fscore': f_score,
        'fscore_nb':f_score_noBg
    }
    return res
