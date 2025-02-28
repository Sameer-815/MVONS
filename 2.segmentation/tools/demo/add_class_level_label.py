import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
import glob
import random
def remove_back():
    # img_name = [name.rstrip('\n') for name in open('F:/guidian/dataset/BCSS_10x/new/train.lst','r').readlines()]
    img_name = glob.glob(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\mask_patch\\' + '*.png'))
    for img in tqdm(img_name):
        mask = cv2.imread(img , cv2.IMREAD_GRAYSCALE)
        name = img.split('\\')[-1][:-4]
        mask = np.array(mask)
        target_count = np.count_nonzero(mask == 4)
        ratio = target_count / (224*224)
        if ratio <= 0.5:
            shutil.copy(img, os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/mask/' + name + '.png'))
            shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/img_patch/' + name + '.png'),
                        os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/img/' + name + '.png'))
def mask_color():
    palette = [0] * 15
    palette[0:3] = [255, 0, 0]  # Tumor (TUM)
    palette[3:6] = [0, 255, 0]  # Stroma (STR)
    palette[6:9] = [0, 0, 255]  # Lymphocytic infiltrate (LYM)
    palette[9:12] = [153, 0, 255]  # Necrosis (NEC)
    palette[12:15] = [255, 255, 255]  # White background or exclude
    img_dir = glob.glob(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\mask\\' , '*.png'))
    for mask in tqdm(img_dir):
        name = mask.split("\\")[-1][:-4]
        mas = Image.open(os.path.join(mask)).convert('P')
        mas.putpalette(palette)
        mas.save(os.path.join('F:\\guidian\\dataset\\BCSS_10x\\new\\noise_train\\mask_color\\', name + '.png'))

def add_label():
    img_name = [name.rstrip('\n') for name in open('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test.lst','r').readlines()]
    for name in tqdm(img_name):
        label = [0] * 4
        # mask = cv2.imread(os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/mask/' + name + '.png'),
        #                  cv2.IMREAD_GRAYSCALE)
        mask = Image.open(os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/mask/' +
                                       name + '.png')).convert('L')
        mask = np.array(mask)
        if np.any(mask == 76):
            label[0] = 1
        if np.any(mask == 150):
            label[1] = 1
        if np.any(mask == 29):
            label[2] = 1
        if np.any(mask == 75):
            label[3] = 1
        # for i in range(4):
        #     if np.any(mask == i):
        #         label[i] = 1
        # print(name + "   " + str(label))
        new_path = name + '[' + str(label[0]) + str(label[1]) + str(label[2]) + str(label[3]) + ']'
        # print(new_path)
        os.rename(os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/img/' + name + '.png') , os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/img/' + new_path + '.png'))
        os.rename(os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/mask/' + name + '.png') , os.path.join('F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/test/test/mask/' + new_path + '.png'))
def remove_noise_mask():
    img_name = [name.rstrip('\n') for name in open('F:/guidian/dataset/BCSS_10x/new/val.lst','r').readlines()]
    for name in tqdm(img_name):
        filepath = os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_mask_test/' + name + '.png')
        shutil.move(filepath, os.path.join('F:/guidian/dataset/BCSS_10x/new/val/noise_mask/' + name + '.png'))
def make_dataset():
    # 获取所有图片的文件名列表
    image_list = os.listdir('F:/guidian/dataset/BCSS_10x/new/noise_train/img/')

    # 定义训练集、验证集和测试集的比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 计算训练集、验证集和测试集的图片数量
    num_images = len(image_list)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val

    # 打乱图片顺序
    random.shuffle(image_list)

    # 分配图片到训练集、验证集和测试集
    train_images = image_list[:num_train]
    val_images = image_list[num_train:num_train + num_val]
    test_images = image_list[-num_test:]

    # 移动图片到对应的文件夹
    for image in train_images:
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/img/' + image), os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/train/img/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/mask_color/' + image), os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/train/mask/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/noise_mask/' + image), os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/train/noise_mask/' + image))

    with open('F:/guidian/dataset/BCSS_10x/new/noise_train/train.lst', 'w') as f:
        for file_name in train_images:
            f.write(file_name[:-4] + '\n')
    for image in val_images:
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/img/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/val/img/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/mask_color/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/val/mask/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/noise_mask/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/val/noise_mask/' + image))
    with open('F:/guidian/dataset/BCSS_10x/new/noise_train/val.lst', 'w') as f:
        for file_name in val_images:
            f.write(file_name[:-4] + '\n')
    for image in test_images:
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/img/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/test/img/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/mask_color/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/test/mask/' + image))
        shutil.copy(os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/noise_mask/' + image),
                    os.path.join('F:/guidian/dataset/BCSS_10x/new/noise_train/test/noise_mask/' + image))
    with open('F:/guidian/dataset/BCSS_10x/new/noise_train/test.lst', 'w') as f:
        for file_name in test_images:
            f.write(file_name[:-4] + '\n')
if __name__ == '__main__':
    add_label()