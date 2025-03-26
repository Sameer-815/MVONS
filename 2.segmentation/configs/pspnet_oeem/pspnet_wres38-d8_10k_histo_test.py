_base_ = './pspnet_wres38-d8_10k_histo.py'
test_cfg=dict(mode='whole', crop_size=(224, 224),crf=False,pred_out_path='F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/WSI/wsi_cut/405723/3200_2240/pred/')
#bash tools/dist_test.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo_test.py G:/train_pth/luad_vote_mask_0.75_1_1.25_1212/iter_20000.pth 1

