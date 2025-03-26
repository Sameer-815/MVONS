_base_ = './pspnet_wres38-d8_10k_histo.py'
test_cfg=dict(mode='whole', crop_size=(224, 224),crf=False,pred_out_path='F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/WSI/wsi_cut/405723/3200_2240/pred2/')
