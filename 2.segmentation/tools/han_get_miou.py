from tools.metrics import Evaluator
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
evaluator = Evaluator(4)
evaluator.reset()
#luad Tumor Necrosis Lymphocyte stroma
# pred_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\multi_scale\\gradcam++\\'
# pred_path = 'F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\test\\test\\grad++_mask\\'
pred_path = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\luad_gradcampp_onss_0814\\mask_refine\\'
# pred_path = 'F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\luad_gradcampp_onss_0814\\mask_gray\\'
# pred_path = 'G:\\train_pth\\bcss_vote_mask_0.75_1_1.25_0102\\mask\\'
# gt_path = 'F:\\guidian\\dataset\\BCSS_10x\\new\\test\\mask\\'
gt_path = 'F:\\data\\data_all\\weak_suprvised_data\\LUAD-HistoSeg\\LUAD-HistoSeg\\test\\mask\\'
# gt_path = 'F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\test\\mask\\'
# gt_path = 'F:\\data\\data_all\\weak_suprvised_data\\BCSS-WSSS\\BCSS-WSSS\\test\\test_tmp\\mask\\'
pred_list = glob.glob(os.path.join(pred_path + '*.png'))
for _pred in tqdm(pred_list):
    png_name = _pred.split('\\')[-1][:-4]
    pred = Image.open(_pred)
    gt = Image.open(os.path.join(gt_path + png_name + '.png'))
    pred = np.expand_dims(np.array(pred),axis=0)
    gt = np.expand_dims(np.array(gt),axis=0)
    # print(pred)
    pred[gt==4]=4
    '''change class'''
    pred_tmp = np.copy(pred)
    # pred_tmp[pred==1]=3
    # pred_tmp[pred==3]=1
    pred_png = Image.fromarray(np.uint8(pred[0]))
    # pred_png.save(os.path.join("F:\\code\\weakly-supervised\\OEEM-main\\segmentation\\runs\\bcss_gradcampp_wo_onss_0926\\mask_gray_refine\\",str(png_name) + '.png'))
    # pred = np.expand_dims(pred,axis=0)

    # print(pred.shape)
    # print(gt.shape)
    evaluator.add_batch(gt,pred)
Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
ious = evaluator.Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
print('IoUs: ', ious)