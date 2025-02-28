import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
img_path = glob.glob(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\images\\','*.png'))
save_path = 'F:\\guidian\\dataset\\BCSS_10x\\img_normalized\\'
for imgs in tqdm(img_path):
    name = imgs.split("\\")[-1][:-4]
    img = cv2.imread(imgs)
    img = img.astype(np.float32)
    normalized_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_image = normalized_image.astype(np.uint8)
    cv2.imwrite(os.path.join(save_path + name + '.png'),normalized_image)
