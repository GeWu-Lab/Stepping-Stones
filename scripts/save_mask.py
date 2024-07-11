import numpy as np
import json
from PIL import Image

def getpallete(num_cls=71):
    """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
    71 is the total category number of V2 dataset, you should not change that"""
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete  # list, lenth is n_classes*3

def convert_matrix_to_rgb_array(matrix, colors):
    matrix_np = matrix.numpy()
    rgb_array = np.zeros((3, matrix_np.shape[0], matrix_np.shape[1]), dtype=np.uint8)
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            rgb_array[:, i, j] = colors[matrix_np[i, j]]

    return rgb_array

def save_batch_raw_mask(name, batch_torch_mask,args):
    batch_torch_mask = batch_torch_mask.cpu()
    batch = len(batch_torch_mask)
    with open(f'{args.data_path}/label2idx.json', 'r') as fr:
        v2_json = json.load(fr)
    v2_pallete = getpallete(num_cls=71) # list
    result = []
    for i in v2_json.values():
        result.append(v2_pallete[(int(i)-1)*3:(int(i)-1)*3+3])
    array_color = np.array(result)
    for i in range(batch):
        mask = convert_matrix_to_rgb_array(batch_torch_mask[i], array_color)
        image = Image.fromarray(np.transpose(mask, (1, 2, 0)))
        image.save(f'{args.ckpt_dir}/save_masks/{name}_{str(i)}.png')

