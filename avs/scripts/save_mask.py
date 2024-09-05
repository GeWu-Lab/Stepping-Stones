import cv2,os

def save_batch_mask(logits,save_path):
    mask=logits.argmax(1)
    for i in range(mask.shape[0]):
        cv2.imwrite(os.path.join(save_path,f"{i}.png"),mask[i].numpy()*255)

