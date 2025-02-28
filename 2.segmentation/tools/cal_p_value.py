from tools.metrics import Evaluator
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
evaluator = Evaluator(4)
# pred_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\han_mask\\'
# gt_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\mask\\'
# miou_han = []
# pred_list = glob.glob(os.path.join(pred_path + '*.png'))
# for _pred in tqdm(pred_list):
#     evaluator.reset()
#     png_name = _pred.split('\\')[-1][:-4]
#     pred = Image.open(_pred)
#     gt = Image.open(os.path.join(gt_path + png_name + '.png'))
#     pred = np.expand_dims(np.array(pred),axis=0)
#     gt = np.expand_dims(np.array(gt),axis=0)
#     # print(pred)
#     pred[gt==4]=4
#     '''change class'''
#     pred_tmp = np.copy(pred)
#     pred_tmp[pred==1]=3
#     pred_tmp[pred==3]=1
#     pred_png = Image.fromarray(np.uint8(pred[0]))
#     # pred_png.save(os.path.join("F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\bcss_gradcampp_wo_onss_0926\\mask_gray_refine\\",str(png_name) + '.png'))
#     # pred = np.expand_dims(pred,axis=0)
#
#     # print(pred.shape)
#     # print(gt.shape)
#     evaluator.add_batch(gt,pred)
#     Acc = evaluator.Pixel_Accuracy()
#     Acc_class = evaluator.Pixel_Accuracy_Class()
#     mIoU = evaluator.Mean_Intersection_over_Union()
#     ious = evaluator.Intersection_over_Union()
#     FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
#     miou_han.append(Acc)
# # pred_path = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\luad_gradcampp_onss_0814\\mask_gray\\'
# pred_path = 'G:\\train_pth\\luad_vote_mask_0.75_1_1.25_1212\\iter_20000\\mask\\'
#
# miou_ours = []
# pred_list = glob.glob(os.path.join(pred_path + '*.png'))
# for _pred in tqdm(pred_list):
#     evaluator.reset()
#     png_name = _pred.split('\\')[-1][:-4]
#     pred = Image.open(_pred)
#     gt = Image.open(os.path.join(gt_path + png_name + '.png'))
#     pred = np.expand_dims(np.array(pred),axis=0)
#     gt = np.expand_dims(np.array(gt),axis=0)
#     pred[gt==4]=4
#     '''change class'''
#     pred_tmp = np.copy(pred)
#     pred_tmp[pred==1]=3
#     pred_tmp[pred==3]=1
#     pred_png = Image.fromarray(np.uint8(pred[0]))
#     evaluator.add_batch(gt,pred)
#     mIoU = evaluator.Mean_Intersection_over_Union()
#     miou_ours.append(Acc)
# t,p = stats.ttest_ind(miou_ours,miou_han)
# a = 0
# for m in miou_ours:
#     a += m;
# print(np.mean(miou_ours))
# mean_han = np.mean(miou_han)
# mean_our = np.mean(miou_ours)
# diff = mean_our - mean_han
# len_data = len(miou_ours)
# se = np.sqrt(((1 - mean_our) * mean_our / len_data) + ((1 - mean_han) * mean_han / len_data))
# t = diff / se
# p = ttest_ind(miou_han, miou_ours, equal_var=False).pvalue / 2  # 单侧检验
# print('mean IoU A:', mean_our)
# print('mean IoU B:', mean_han)
# print('IoU difference:', diff)
# print('p-value:', p)

import numpy as np
from scipy.stats import ttest_ind

def tmp():
    han_miou = []
    our_miou = []
    pred_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\train_PM\\PM_1%\\han_mask_w_bcss_stage1\\'
    gt_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\mask\\'
    pred_list = glob.glob(os.path.join(pred_path + '*.png'))
    for _pred in tqdm(pred_list):
        evaluator.reset()
        png_name = _pred.split('\\')[-1][:-4]
        pred = Image.open(_pred)
        gt = Image.open(os.path.join(gt_path + png_name + '.png'))
        pred = np.expand_dims(np.array(pred), axis=0)
        gt = np.expand_dims(np.array(gt), axis=0)
        pred[gt == 4] = 4
        evaluator.add_batch(gt, pred)
        Acc = evaluator.Pixel_Accuracy()
        han_miou.append(Acc)
        evaluator.reset()
        pred2 = Image.open(os.path.join('G:\\train_pth\\luad_use_bcsspth_0315_1%\\mask\\' + png_name + '.png'))
        pred2 = np.expand_dims(np.array(pred2), axis=0)
        pred2[gt == 4] = 4
        evaluator.add_batch(gt, pred2)
        Acc = evaluator.Pixel_Accuracy()
        our_miou.append(Acc)
    print(han_miou)
    mean_han = np.mean(han_miou)
    mean_our = np.mean(our_miou)
    diff = mean_our - mean_han
    len_data = len(our_miou)
    se = np.sqrt(((1 - mean_our) * mean_our / len_data) + ((1 - mean_han) * mean_han / len_data))
    t = diff / se
    p = ttest_ind(our_miou ,han_miou, equal_var=False).pvalue / 2  # 单侧检验
    print('mean IoU A:', mean_our)
    print('mean IoU B:', mean_han)
    print('IoU difference:', diff)
    print('p-value:', p)
if __name__ == '__main__':
    tmp()