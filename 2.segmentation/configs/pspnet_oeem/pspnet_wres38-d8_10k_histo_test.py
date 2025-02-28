_base_ = './pspnet_wres38-d8_10k_histo.py'

# test_cfg = dict(mode='whole', crop_size=(320, 320), stride=(256, 256), crf=False,
#                 pred_output_path='glas', npy=True)
# test_cfg = dict(mode='whole',crf=False,pred_out_path='G:/baidu_disk/WSI_seg/00403517 202017124HE/img/mask_color/',npy=True)
# test_cfg = dict(mode='slide', crop_size=(112, 112), stride=(112, 112),crf=False,pred_out_path='F:/code/weakly-supervised/OEEM-main/segmentation/runs/bcss_gradcam_onss_0923/pred/',npy=True)
test_cfg=dict(mode='whole', crop_size=(224, 224),crf=False,pred_out_path='F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/WSI/wsi_cut/405723/3200_2240/pred2/')
#bash tools/dist_test.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo_test.py G:/train_pth/luad_vote_mask_0.75_1_1.25_1212/iter_20000.pth 1

