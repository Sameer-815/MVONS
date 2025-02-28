import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from tool.GenDataset import make_pred_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
from PIL import Image

class Deeplab(object):
    def __init__(self,args):
        self.args = args
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.pred_loader = make_pred_loader(args, **kwargs)
        self.nclass = args.n_class
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                               args.epochs, len(self.train_loader))
        import importlib
        model_stage1 = getattr(importlib.import_module('network.resnet38_cls'), 'Net_CAM')(n_class=4)
        resume_stage1 = 'checkpoints/stage1_checkpoint_trained_on_luad'  + '.pth'
        weights_dict = torch.load(resume_stage1)
        model_stage1.load_state_dict(weights_dict)
        self.model_stage1 = model_stage1.cuda()
        self.model_stage1.eval()
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        # args.resume = 'init_weights/deeplab-resnet.pth.tar'
        # args.resume = 'checkpoints/stage2_checkpoint_trained_on_' + str(args.dataset) + '.pth'
        # args.resume = 'checkpoints/bcss_use_luadpth/bcss_use_luadpth_0322_luad10%.pth'
        # args.resume = 'checkpoints/stage2_checkpoint_trained_on_luad.pth'
        checkpoint = torch.load(args.resume)
        W = checkpoint['state_dict']
        self.model.module.load_state_dict(W, strict=False)
        print("=> loaded checkpoint '{}' ".format(args.resume))
    def detect_image(self,Is_GM):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.pred_loader, desc='\r')
        # print(self.test_loader.dataset)
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            png_name = sample[1][0]
            png_name = png_name.split('/')[-1][:-4]
            # print(png_name)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    _, y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    # print(y_cls)
                    pred_cls = (y_cls > 0.1)
                    # print(pred_cls)
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred * (pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target == 4] = 4
            pred_png = Image.fromarray(np.uint8(pred[0]))
            pred_png.save(os.path.join("F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/test/han_mask/",
                                       str(png_name) + '.png'))
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Pred:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

def main():
        parser = argparse.ArgumentParser(description="WSSS Stage2")
        parser.add_argument('--backbone', type=str, default='resnet',
                            choices=['resnet', 'xception', 'drn', 'mobilenet'])
        parser.add_argument('--out-stride', type=int, default=16)
        parser.add_argument('--Is_GM', type=bool, default=True, help='Enable the Gate mechanism in test phase')
        parser.add_argument('--dataroot', type=str,
                            default='F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/')
        # parser.add_argument('--dataroot', type=str,
        #                     default='F:/guidian/dataset/BCSS_10x/new/')
        parser.add_argument('--dataset', type=str, default='luad')
        parser.add_argument('--savepath', type=str, default='checkpoints/')
        parser.add_argument('--workers', type=int, default=10, metavar='N')
        parser.add_argument('--sync-bn', type=bool, default=None)
        parser.add_argument('--freeze-bn', type=bool, default=False)
        parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
        parser.add_argument('--n_class', type=int, default=4)
        # training hyper params
        parser.add_argument('--epochs', type=int, default=20, metavar='N')
        parser.add_argument('--batch-size', type=int, default=20, metavar='N')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
        parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
        parser.add_argument('--nesterov', action='store_true', default=False)
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=False)
        parser.add_argument('--gpu-ids', type=str, default='0')
        parser.add_argument('--seed', type=int, default=1, metavar='S')
        # checking point
        parser.add_argument('--resume', type=str, default='checkpoints/stage2_checkpoint_trained_on_luad.pth')
        parser.add_argument('--checkname', type=str, default='deeplab-resnet')
        parser.add_argument('--ft', action='store_true', default=False)
        parser.add_argument('--eval-interval', type=int, default=1)
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            try:
                args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if args.sync_bn is None:
            if args.cuda and len(args.gpu_ids) > 1:
                args.sync_bn = True
            else:
                args.sync_bn = False
        print(args)
        deeplab = Deeplab(args)
        deeplab.detect_image(args.Is_GM)
if __name__ == '__main__':
    main()
