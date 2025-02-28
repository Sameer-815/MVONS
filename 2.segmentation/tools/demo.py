from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from skimage import morphology
import cv2
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm
def wsi_seg(img_dir):
    config_file = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\configs\\pspnet_oeem\\pspnet_wres38-d8_10k_histo_test.py'
    checkpoint_file = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\luad_gradcampp_onss_0814\\latest.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    # img = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\img\\387709-10004-55944[1 0 0 1].png'
    # img = 'F:\\data\\shengyi\\wsi\\tmp\\405723.png'
    result = inference_segmentor(model, img_dir)
    show_result_pyplot(model, img_dir, result, get_palette('wsssluad'))
def wsi_seg_bcss(img_dir):
    config_file = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\configs\\pspnet_oeem\\pspnet_wres38-d8_10k_histo_test.py'
    checkpoint_file = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\bcss_gradcam_onss_0923\\best_iter_18000.pth'
    model = init_segmentor(config_file,checkpoint_file,device='cuda:0')
    result = inference_segmentor(model,img_dir)
    show_result_pyplot(model,img_dir,result,get_palette('wsssbcss'))

def gen_bg_mask( img_dir):
    orig_img = Image.open(img_dir).convert('RGB')
    orig_img = np.asarray(orig_img)
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    binary = np.uint8(binary)
    # dst = morphology.remove_small_objects(binary != 255, min_size=10000, connectivity=1)
    # dst = morphology.remove_small_objects(dst == False, min_size=10000, connectivity=1)
    # bg_mask = np.ones(orig_img.shape[:2]) * -10000
    # bg_mask[dst == True] = 10000
    return binary

if __name__ == '__main__':
    base_dir = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\WSI\\wsi_cut\\405723\\3200_2240\\'
    img_dirs = glob.glob(os.path.join(base_dir, "patch", "*.png"))
    data = 'luad'
    if data == 'luad':
        for img_dir in tqdm(img_dirs):
            palette = [0] * 15  # LUAD
            palette[0:3] = [205, 51, 51]
            palette[3:6] = [0, 255, 0]
            palette[6:9] = [65, 105, 225]
            palette[9:12] = [255, 165, 0]
            palette[12:15] = [255, 255, 255]
            wsi_seg(img_dir)
            png_name = img_dir.split('\\')[-1][:-4]
            bg_mask = gen_bg_mask(os.path.join(base_dir, "patch\\" +  str(png_name) + '.png'))
            mask = cv2.imread(os.path.join(base_dir, "pred2\\mask\\" + str(png_name) + '.png'))
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    gray = bg_mask[i, j]
                    if gray==255:
                        mask[i, j] = np.uint8(4)
            cv2.imwrite(os.path.join(base_dir, "pred2\\mask\\" + png_name + '.png'), mask)
            mask_gray = cv2.imread(os.path.join(
                base_dir, "pred2\\mask\\",
                png_name + '.png'))
            mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(
                base_dir, "pred2\\mask\\",
                png_name + '.png'), mask_gray)
            mas = Image.open(os.path.join(base_dir, "pred2\\mask\\" + png_name + '.png'))
            mas.convert('P')
            # visualimg  = Image.fromarray(mas.astype(np.uint8), "P")
            mas.putpalette(palette)
            mas.save(os.path.join(base_dir, "pred2\\mask_color\\" + png_name + '.png'), format='PNG')
            img = cv2.imread(os.path.join(img_dir))
            mask_color = cv2.imread(os.path.join(base_dir, "pred2\\mask_color\\", png_name + '.png'))
            dst = cv2.addWeighted(mask_color, 0.3, img, 0.7, 0)
            cv2.imwrite(os.path.join(base_dir, "pred2\\overlay\\"+png_name+'.png'),dst)
            # img = cv2.imread(os.path.join("F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\wsi_cut\\405723\\3200_2240\\pred\\mask_color_new\\", png_name + '.png'))
            # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            # cv2.imwrite(os.path.join("F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\wsi_cut\\405723\\3200_2240\\pred\\mask_color_new\\", png_name + '.png'), img)
    if data == 'luad_patch':
        for img_dir in tqdm(img_dirs):
            wsi_seg(img_dir)
    if data == 'bcss':
        palette = [0] * 15  # LUAD
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
        for img_dir in tqdm(img_dirs):
            wsi_seg_bcss(img_dir)
            png_name = img_dir.split('\\')[-1][:-4]
            bg_mask = cv2.imread(os.path.join('F:\\guidian\\dataset\\BCSS_40X\\mask5\\' + png_name + '.png'))
            bg_mask = cv2.cvtColor(bg_mask,cv2.COLOR_RGB2GRAY)
            mask = cv2.imread(os.path.join('F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask\\' + png_name + '.png'))
            print(mask.shape[0])
            print(mask.shape[1])
            height= mask.shape[0]
            width = mask.shape[1]
            for i in range(height):
                for j in range(width):
                    gray = bg_mask[i,j]
                    if gray == 0:
                        mask[i,j] = np.uint8(4)
            cv2.imwrite(os.path.join('F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask\\' + png_name + '.png'),mask)
            mask = cv2.imread(os.path.join('F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask\\' + png_name + '.png'))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join("F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask\\",
                png_name + '.png'), mask)
            mas = Image.open(os.path.join("F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask\\" + png_name + '.png'))
            mas.convert('P')
            mas.putpalette(palette)
            mas.save(os.path.join(
                "F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask_color\\",png_name + '.png'), format='PNG')
            img = cv2.imread(os.path.join(img_dir))
            mask_color = cv2.imread(os.path.join(
                "F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\mask_color\\",
                png_name + '.png'))
            dst = cv2.addWeighted(mask_color, 0.3, img, 0.7, 0)
            cv2.imwrite(os.path.join(
                "F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\WSI\\pred\\overlay\\" + png_name + '.png'),
                        dst)