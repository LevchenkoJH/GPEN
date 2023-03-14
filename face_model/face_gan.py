'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import torch
import os
import cv2
import glob
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, utils
from gpen_model import FullGenerator, FullGenerator_SR

class FaceGAN(object):
    def __init__(self, base_dir='./', in_size=512, out_size=None, model=None, channel_multiplier=2, narrow=1, key=None, is_norm=True, device='cuda'):
        self.mfile = os.path.join(base_dir, 'weights', model+'.pth')
        self.n_mlp = 8
        self.device = device
        self.is_norm = is_norm
        self.in_resolution = in_size
        self.out_resolution = in_size if out_size is None else out_size
        self.key = key
        self.load_model(channel_multiplier, narrow)

    def load_model(self, channel_multiplier=2, narrow=1):
        if self.in_resolution == self.out_resolution:
            print("load_modal")
            self.model = FullGenerator(size=self.in_resolution, style_dim=512, n_mlp=self.n_mlp, channel_multiplier=channel_multiplier, narrow=narrow, device=self.device)
            # self.model = FullGenerator(self.in_resolution, self.in_resolution, self.n_mlp, channel_multiplier, narrow=narrow, device=self.device)
        else:
            self.model = FullGenerator_SR(self.in_resolution, self.out_resolution, 512, self.n_mlp, channel_multiplier, narrow=narrow, device=self.device)
        print(self.mfile)
        pretrained_dict = torch.load(self.mfile, map_location=torch.device('cpu'))
        if self.key is not None: pretrained_dict = pretrained_dict[self.key]
        # ВРЕМЕННО!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.model.load_state_dict(pretrained_dict['g'])
        self.model.to(self.device)
        self.model.eval()

    def process(self, corr_img, img):

        # Корреляционное изображение такогоже размера как и восстанавливаемое
        # corr_img = cv2.resize(corr_img, (self.in_resolution, self.in_resolution))
        img = cv2.resize(img, (self.in_resolution, self.in_resolution))

        # corr_img_t = self.img2tensor(corr_img)
        img_t = self.img2tensor(img)


        with torch.no_grad():
            # generator(degraded_img, correlation_features)
            # Модель - генератор

            print("corr_img_t -> ", corr_img.shape)
            print("img_t", img_t.shape)

            out, __ = self.model(img_t, corr_img)
        # Удаляем за ненадобностью
        # del  corr_img_t
        del img_t

        # Восстановленный кадр
        out = self.tensor2img(out)

        return out

    def img2tensor(self, img):
        img_t = torch.from_numpy(img).to(self.device)/255.
        if self.is_norm:
            img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1) # BGR->RGB
        return img_t

    def tensor2img(self, img_t, pmax=255.0, imtype=np.uint8):
        if self.is_norm:
            img_t = img_t * 0.5 + 0.5
        img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * pmax

        return img_np.astype(imtype)
