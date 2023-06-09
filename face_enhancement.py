'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import cv2
import time
import numpy as np
import __init_paths
import clip
import torch

from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points
from PIL import Image

class FaceEnhancement(object):
    def __init__(self, args, base_dir='./', in_size=512, out_size=None, model=None, use_sr=True, device='cuda'):

        self.facedetector = RetinaFaceDetection(base_dir, device)

        self.facegan = FaceGAN(base_dir, in_size, out_size, model, args.channel_multiplier, args.narrow, args.key, device=device)

        self.srmodel =  RealESRNet(base_dir, args.sr_model, args.sr_scale, args.tile_size, device=device)

        self.faceparser = FaceParse(base_dir, device=device)

        self.use_sr = use_sr

        self.in_size = in_size

        self.out_size = in_size if out_size is None else out_size

        self.threshold = 0.9

        self.alpha = args.alpha

        # the mask for pasting restored faces back
        # маска для вклейки восстановленных лиц обратно
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        # получить опорную позицию 5 ориентиров в настройках обрезки
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.in_size, self.in_size), inner_padding_factor, outer_padding, default_square)

        self.clip_model, self.clip_preprocess = clip.load('RN50', device)














    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def TEST_cicle(self, img, facebs, landms, orig_faces, enhanced_faces):
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold: continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts,
                                             crop_size=(self.in_size, self.in_size))

            # enhance the face
            ef = self.facegan.process(of)

            orig_faces.append(of)
            enhanced_faces.append(ef)

            # tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]
        return full_mask, full_img










    def get_clip_image_features(self, correlation_img, clip_model, clip_preprocess):

        correlation_img = np.expand_dims(correlation_img, 0)
        # print("correlation_img ->", correlation_img.shape)

        # Обратные проеобразования для правильной формы входа в CLIP
        correlation_img_copy = correlation_img
        correlation_img_copy = (correlation_img_copy * 0.5) + 0.5
        correlation_img_copy = (correlation_img_copy * 255.).astype(np.uint8)
        # correlation_img_copy = correlation_img_copy.transpose(0, 2, 3, 1)
        # print("correlation_img_copy ->", correlation_img_copy.shape)

        with torch.no_grad():
            # clip_preprocess не поддерживает батчинг
            clip_image_inputs = []
            for c_img in correlation_img_copy:
                # print("c_img ->", c_img.shape)
                pil_img = Image.fromarray(c_img)

                # Prepare the inputs
                if len(clip_image_inputs) == 0:
                    clip_image_inputs = clip_preprocess(pil_img).unsqueeze(0)
                else:
                    clip_image_inputs = torch.cat((clip_image_inputs, clip_preprocess(pil_img).unsqueeze(0)))



            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # clip_image_inputs = clip_image_inputs.to('cuda')




            # Calculate features
            clip_image_features = clip_model.encode_image(clip_image_inputs)
            # print(clip_image_features.shape)
            # print(type(clip_image_features))
            # print(clip_image_features)
            return clip_image_features.to(torch.float32)



    def process(self, img, corr_img, isFirst=False, aligned=False):
        # Для отчетности
        orig_faces, enhanced_faces = [], []

        # Если изображения выровнены
        if aligned:

            # aligned False
            print("aligned", aligned)
            # FaceGan теперь принимает на вход 2 изображения
            # Нужно изменить внутренности его process
            # Не нужно дублировать corr
            ef = self.facegan.process(corr_img=corr_img, img=img)


            # Для отчетности
            orig_faces.append(img)
            enhanced_faces.append(ef)



            # При увеличении разрешения
            if self.use_sr:
                # Коррелируемый кадр не нужно увеличивать
                ef = self.srmodel.process(ef)

            # Коррелируемый кадр не возвращаем
            return ef, orig_faces, enhanced_faces


        # aligned False
        # print("aligned", aligned)
        # При увеличении разрешения
        if self.use_sr:

            # print("self.use_sr", self.use_sr)


            # Повторяем манипуляции с коррелируемым кадром
            # Увеличиваем его разрешение
            corr_sr = self.srmodel.process(corr_img)
            img_sr = self.srmodel.process(img)

            # Изменяем shape
            if corr_sr is not None:
                corr_img = cv2.resize(corr_img, corr_sr.shape[:2][::-1])
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])


        # Находим лицо на коррелируемом кадре
        corr_facebs, corr_landms = self.facedetector.detect(corr_img)
        facebs, landms = self.facedetector.detect(img)

        # print("corr_facebs", corr_facebs)
        # print("corr_landms", corr_landms)
        #
        # print("facebs", facebs)
        # print("landms", landms)


        # self.use_sr False
        # print("self.use_sr", self.use_sr)




        # corr_full_mask, corr_full_img = self.TEST_cicle(img=corr_img, facebs=corr_facebs, landms=corr_landms, orig_faces=[], enhanced_faces=[])
        # full_mask, full_img = self.TEST_cicle(img=img, facebs=facebs, landms=landms, orig_faces=orig_faces, enhanced_faces=enhanced_faces)








        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        # full_mask(450, 450)
        # print("full_mask", full_mask.shape)


        # full_img (450, 450, 3)
        # print("full_img", full_img.shape)


        # for i, (faceb, facial5points, corr_faceb, corr_facial5points) in enumerate(zip(facebs, landms, corr_facebs, corr_landms)):
        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):

            # print(faceb, facial5points)#, corr_faceb, corr_facial5points)


            # Нужно учесть что коррелируемое изображение может быть пустым

            # Слабое место
            # Пусть вырезы у обоих кадров будут одинаковыми (на основе восстанавливаемого)
            if faceb[4] < self.threshold: continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])


            facial5points = np.reshape(facial5points, (2, 5))
            # corr_facial5points = np.reshape(corr_facial5points, (2, 5))




            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts,
                                             crop_size=(self.in_size, self.in_size))
            # corr_of, corr_tfm_inv = warp_and_crop_face(corr_img, corr_facial5points, reference_pts=self.reference_5pts,
            #                                  crop_size=(self.in_size, self.in_size))

            corr_of, corr_tfm_inv = warp_and_crop_face(corr_img, facial5points, reference_pts=self.reference_5pts,
                                                       crop_size=(self.in_size, self.in_size))






            # В место коррелируемого кадра мы теперь хотим отправлять его признаки подсчитаныне с помощью CLIP
            correlation_features = self.get_clip_image_features(corr_of, self.clip_model, self.clip_preprocess)


            if isFirst:
                # print("FIRST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                correlation_features = torch.zeros_like(correlation_features)



            # print("correlation_features -> ->", correlation_features)

            # enhance the face
            ef = self.facegan.process(corr_img=correlation_features, img=of)





            orig_faces.append(of)
            enhanced_faces.append(ef)

            # tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]























        full_mask = full_mask[:, :, np.newaxis]
        # corr_full_mask = corr_full_mask[:, :, np.newaxis]

        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
            # corr_img = cv2.convertScaleAbs(corr_sr * (1 - corr_full_mask) + corr_full_img * corr_full_mask)
        else:
            img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)
            # corr_img = cv2.convertScaleAbs(corr_img * (1 - corr_full_mask) + corr_full_img * corr_full_mask)

        return img, orig_faces, enhanced_faces
        
        
