from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
mask_path = glob.glob(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\masks\\' , '*.png'))
save_path = 'F:\\guidian\\dataset\\BCSS_10x\\newMask\\'
palette = [0]*15
palette[0:3] = [255, 0, 0]          # Tumor (TUM)
palette[3:6] = [0,255,0]            # Stroma (STR)
palette[6:9] = [0,0,255]            # Lymphocytic infiltrate (LYM)
palette[9:12] = [153, 0, 255]       # Necrosis (NEC)
palette[12:15] = [255, 255, 255]    # White background or exclude
for masks in tqdm(mask_path):
    name = masks.split("\\")[-1][:-4]
    mask = np.array(Image.open(masks))
    newMask = np.empty((mask.shape[0],mask.shape[1]),dtype=np.int32)
    newMask[mask==1]=0
    newMask[mask==2]=1
    newMask[mask==3]=2
    newMask[mask==4]=3
    newMask[mask==0]=4
    newMask[mask>4]=4
    newMask = Image.fromarray(newMask)
    # newMask.putpalette(palette)
    newMask.save(os.path.join(save_path, name + '.png'), format='PNG')
    mas = Image.open(os.path.join(save_path + name + '.png')).convert('P')
    mas.putpalette(palette)
    mas.save(os.path.join(save_path, name + '.png'))

