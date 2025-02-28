import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import glob
import random
def make_noise():
    palette = [0]*15
    palette[0:3] = [255, 0, 0]          # Tumor (TUM)
    palette[3:6] = [0,255,0]            # Stroma (STR)
    palette[6:9] = [0,0,255]            # Lymphocytic infiltrate (LYM)
    palette[9:12] = [153, 0, 255]       # Necrosis (NEC)
    palette[12:15] = [255, 255, 255]    # White background or exclude
    # Read in the binary image

    img_dir = glob.glob(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\mask\\', '*.png'))
    for imgs in tqdm(img_dir):
        name = imgs.split('\\')[-1][:-4]
        img = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)
        # Define the structuring element for dilation and erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #随机选择一种方式
        rand = random.choice([0, 1])
        # Apply dilation to simulate over-annotation
        if rand == 0:
            dilated = cv2.dilate(img, kernel, iterations=2)
            dilated = Image.fromarray(dilated)
            dilated.save(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png'), format='PNG')
            mas = Image.open(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png')).convert('P')
            mas.putpalette(palette)
            mas.save(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png'))
        else:
        # Apply erosion to simulate under-annotation
            eroded = cv2.erode(img, kernel, iterations=2)
            eroded = Image.fromarray(eroded)
            eroded.save(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png'), format='PNG')
            mas = Image.open(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png')).convert('P')
            mas.putpalette(palette)
            mas.save(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\noise_mask\\' + name + '.png'))
if __name__ == '__main__':
    make_noise()