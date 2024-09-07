import numpy as np
import torch
import cv2
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
    # 将torch矩阵转换为NumPy数组
    matrix_np = matrix.numpy()

    # 创建形状为(3, w, h)的空NumPy数组
    rgb_array = np.zeros((3, matrix_np.shape[0], matrix_np.shape[1]), dtype=np.uint8)
    # 使用对应位置的三维数组代替矩阵中的整数
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            rgb_array[:, i, j] = colors[matrix_np[i, j]]
    # for i in range(3):
    #     print([matrix_np])
    #     rgb_array[i] = colors[matrix_np]

    return rgb_array

def convert_list_to_numpy_array(lst):
    # 将列表转换为NumPy数组
    array = np.array(lst)

    # 调整数组形状为(71, 3)
    reshaped_array = array.reshape((71, 3))

    return reshaped_array

def save_mask(name,rgb_array):
    print(rgb_array.shape,)
    # bgr_array = np.transpose(rgb_array, (1, 2, 0))
    # print(rgb_array.shape)
    image = Image.fromarray(np.transpose(rgb_array, (1, 2, 0)))

    # 保存图像
    image.save(name)
    # cv2.imwrite(name, rgb_array)
    # print(rgb_array.shape,name,np.max(rgb_array))

def raw_mask(name,torch_mask):
    
    with open("/data/users/juncheng_ma/AVSBench-semantic/label2idx.json", 'r') as fr:
        v2_json = json.load(fr)
    # with open("/data/users/peiwen_sun/project/AVS_v2_ft_cpu/datasets_avss/v1m_idx.json", 'r') as fr:
    #     v2_json = json.load(fr)
    # with open(label_to_idx_path, 'r') as fr:
    #     label_to_pallete_idx = json.load(fr)
    v2_pallete = getpallete(num_cls=71) # list
    # result = [[0,0,0]]
    result = []
    for i in v2_json.values():
        result.append(v2_pallete[(int(i)-1)*3:(int(i)-1)*3+3])
    # result = result[1:] + [result[0]]
    array_color = np.array(result)
    # print(array_color)
    # array_color = convert_list_to_numpy_array(result)
    mask = convert_matrix_to_rgb_array(torch_mask, array_color)
    save_mask(name, mask)

def save_batch_raw_mask(name, batch_torch_mask):
    batch_torch_mask = batch_torch_mask.cpu()
    batch = len(batch_torch_mask)
    for i in range(batch):
        raw_mask(name+"_"+str(i)+".jpg", batch_torch_mask[i])


if __name__ == '__main__':
    matrix = torch.randint(0, 71, (720, 1280))
    raw_mask(matrix)