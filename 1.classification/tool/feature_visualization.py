import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
import importlib
from tool import infer_utils
from PIL import Image
import torch.nn.functional as F
from tool.GenDataset import Stage1_InferDataset
from torch.utils.data import DataLoader
import os


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        # self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name)

            if module == self.feature_module:
                # target_activations, x = self.feature_extractor(x)
                x = module(x)
                if name == "bn7":
                    x = F.relu(x)
                x.register_hook(self.save_gradient)
                target_activations += [x]

            else:
                x = module(x)
                if name == "fc8":
                    x = x.view(x.size(0), -1)
                if name == "bn7":
                    x = F.relu(x)

        return target_activations, x  # 指定层的输出和网络最终的输出

class GradCamPlusPlus:
    '''
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    '''

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):

        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)  # 指定层的输出和网络最终的输出
        features = features[-1].cpu().data.numpy().squeeze()
        feature_map_num = features.shape[0]
        print(features[-1])
        for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
            feature = features[index]
            feature = np.asarray(feature * 255, dtype=np.uint8)
            feature = cv2.resize(feature, (224, 224), interpolation=cv2.INTER_NEAREST)  # 改变特征呢图尺寸
            feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)  # 变成伪彩图
            cv2.imwrite('F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/test_feature/channel_{}.png'.format(str(index)), feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='../checkpoints/stage1_checkpoint_trained_on_luad.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/", type=str)
    parser.add_argument("--dataset", default="luad", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--n_class", default=4, type=int)
    args = parser.parse_args()
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    ffmm = model.bn7
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(args.dataroot, 'test/'), transform=transform)  # 111
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(os.path.join(args.dataroot, 'test'), img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCamPlusPlus(model=model, feature_module=ffmm, \
                                   target_layer_names=["1"], use_cuda=True)
        grayscale_cam, _ = grad_cam(img_list, 0)

