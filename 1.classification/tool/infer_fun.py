import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam
from scipy.ndimage import zoom
from tool.gradcamplus import GradCamPlusPlus
from tool.gradcam import GradCam
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img
def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
    return Combine

def infer(model, dataroot, n_class):
    model.eval()
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'img/'),transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]

        img_path = os.path.join(dataroot + 'img/' + img_name+'.png')
        # print(img_path)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.25):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam, y = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                    return cam, label

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam))

        # cam --> segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        '''save_segmap'''
        # palette = [0] * 15
        # palette[0:3] = [205, 51, 51]
        # palette[3:6] = [0, 255, 0]
        # palette[6:9] = [65, 105, 225]
        # palette[9:12] = [255, 165, 0]
        # palette[12:15] = [255, 255, 255]
        # visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        # visualimg.putpalette(palette)
        # visualimg.save(os.path.join('F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/multi_scale/forward_cam_no_vote/', img_name + '.png'), format='PNG')
        if iter%100==0:
            print(iter)
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot,'mask/'), img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

      
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset):
    # print(model)
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(dataset)
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'test/img/'),transform=transform)#111
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0].split('\\')[-1]
        img_path = os.path.join(os.path.join(dataroot,'test/img/'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCam(model=model, feature_module=ffmm, \
                target_layer_names=["1"], use_cuda=True)
        # grad_cam_bn7 = GradCamPlusPlus(model=model, feature_module=model.bn7, target_layer_names=["1"], use_cuda=True)
        # grad_cam_b5_2 = GradCamPlusPlus(model=model, feature_module=model.b5_2, target_layer_names=["1"], use_cuda=True)
        # grad_cam_b6 = GradCamPlusPlus(model=model, feature_module=model.b6, target_layer_names=["1"], use_cuda=True)
        cam = []
        for i in range(n_class):
            target_category = i
            '''多尺度'''
            # input_size_h = img_list.size()[2]
            # input_size_w = img_list.size()[3]
            # x2 = F.interpolate(img_list,size=(int(input_size_h * 0.75), int(input_size_w * 0.75)), mode='bilinear', align_corners=False)
            # x3 = F.interpolate(img_list, size=(int(input_size_h * 1.25), int(input_size_w * 1.25)), mode='bilinear',
            #                    align_corners=False)
            # x4 = F.interpolate(img_list, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',
            #                    align_corners=False)
            # grayscale_cam1, _ = grad_cam(img_list, target_category)
            # grayscale_cam2, _ = grad_cam(x2, target_category)
            # grayscale_cam3, _ = grad_cam(x3, target_category)
            # grayscale_cam4, _ = grad_cam(x4, target_category)
            # np转tensor
            # grayscale_cam2 = torch.from_numpy(grayscale_cam2).unsqueeze(0).unsqueeze(1)
            # grayscale_cam3 = torch.from_numpy(grayscale_cam3).unsqueeze(0).unsqueeze(1)
            # grayscale_cam4 = torch.from_numpy(grayscale_cam4).unsqueeze(0).unsqueeze(1)
            # cv2.imwrite(os.path.join(
            #     "F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/PM_multi_scale_single_maskbn7/" + 'cam1' + '_' + str(
            #         i) + '.png'), show_cam_on_image(grayscale_cam1))
            # cv2.imwrite(os.path.join(
            #     "F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/PM_multi_scale_single_maskbn7/" + 'cam0.5' + '_' + str(
            #         i) + '.png'), show_cam_on_image(grayscale_cam2))
            # cv2.imwrite(os.path.join(
            #     "F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/PM_multi_scale_single_maskbn7/" + 'cam1.5' + '_' + str(
            #         i) + '.png'), show_cam_on_image(grayscale_cam3))
            # cv2.imwrite(os.path.join(
            #     "F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/PM_multi_scale_single_maskbn7/" + 'cam2' + '_' + str(
            #         i) + '.png'), show_cam_on_image(grayscale_cam4))
            # grayscale_cam2 = F.interpolate(grayscale_cam2, size=[int(224), int(224)], mode='bilinear',
            #                      align_corners=False)
            # grayscale_cam3 = F.interpolate(grayscale_cam3,size=[int(224), int(224)], mode='bilinear',
            #                      align_corners=False)
            # grayscale_cam4 = F.interpolate(grayscale_cam4, size=[int(224), int(224)], mode='bilinear',
            #                      align_corners=False)
            # grayscale_cam2 = grayscale_cam2.squeeze(0).squeeze(0).detach().numpy()
            # grayscale_cam3 = grayscale_cam3.squeeze(0).squeeze(0).detach().numpy()
            # grayscale_cam4 = grayscale_cam4.squeeze(0).squeeze(0).detach().numpy()

            # print(grayscale_cam4.shape)
            # grayscale_cam = (0.85*grayscale_cam1 + 0.05*grayscale_cam2 + 0.05*grayscale_cam3 + 0.05*grayscale_cam4)
            # grayscale_cam = (0.8*grayscale_cam1 + 0.1*grayscale_cam2 + 0.1*grayscale_cam3 )
            '''多层特征融合'''
            # grayscale_cam_bn7, _ = grad_cam_bn7(img_list, target_category)
            # grayscale_cam_b5_2, _ = grad_cam_b5_2(img_list, target_category)
            # grayscale_cam_b6, _ = grad_cam_b6(img_list, target_category)
            # # grayscale_cam4, _ = grad_cam(x4, target_category)
            # grayscale_cam = 0.9*grayscale_cam_bn7 + 0.05*grayscale_cam_b5_2 + 0.05*grayscale_cam_b6
            ''''''
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        norm_cam = (norm_cam - np.min(norm_cam))/_range
        ##  Extract the image-level label from the filename
        ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
        ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
        label_str = img_name.split(']')[0].split('[')[-1]
        if dataset == 'luad':
            label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
        elif dataset == 'bcss':
            label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)

        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score , bg_score), axis=0)
        elif dataset == 'bcss':
            bg_score = np.zeros((1,224,224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)

        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, img_name+'.png'), format='PNG')

        if iter%100==0:           
            print(iter)
def show_cam_on_image(mask):
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_BONE)
    # heatmap = np.float32(heatmap) / 255
    # # cam = heatmap + np.float32(img)
    # cam = heatmap
    # cam = cam / np.max(cam)
    # return np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    # cam = cam / np.max(cam)
    return np.uint8(225*cam)