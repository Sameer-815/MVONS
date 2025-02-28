from PIL import Image,ImageFile
import glob
import os
import numpy as np
from tqdm import tqdm
# f = open("o.txt", 'w+')
MASK_PATH = 'F:\\data\\patch\\'
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
def run():

    mask_folder = "F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\mask\\"
    img_folder = "F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\img\\"
    for mask_all_name in os.listdir(mask_folder):
        list1 = [0] * 5
        print(mask_all_name)
        mask_name = mask_all_name[:-13]
        print(mask_name)
        mask_src = os.path.join(mask_folder , mask_all_name)
        img_src = os.path.join(img_folder , mask_name + '.png')
        mask = Image.open(mask_src)
        width, height = mask.size
        for i in range(width):
            for j in range(height):
                n1 = mask.getpixel((i, j))
                list1[n1] = list1[n1] + 1
        # print(mask_name)
        if(list1[0] > 0):
            mask_name = mask_name + '[' + '1'
        else:
            mask_name = mask_name + '[' + '0'
        if (list1[1] > 0):
            mask_name = mask_name + ' ' + '1'
        else:
            mask_name = mask_name + ' ' + '0'
        if (list1[2] > 0):
            mask_name = mask_name + ' ' + '1'
        else:
            mask_name = mask_name + ' ' + '0'
        if (list1[3] > 0):
            mask_name = mask_name + ' ' + '1' + ']'
        else:
            mask_name = mask_name + ' ' + '0' + ']'
        mask_name = mask_name + '.png'
        os.rename(mask_src , os.path.join(mask_folder , mask_name))
        os.rename(img_src , os.path.join(img_folder , mask_name))
        # print(list1)

if __name__ == '__main__':
    run()