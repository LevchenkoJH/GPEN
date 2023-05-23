'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model import FullGenerator, Discriminator

from training.loss.id_loss import IDLoss
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips

import numpy as np
import clip
from PIL import Image

# shuffle - перемешивать
# distributed - распределенный
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    # print("--------------------------------------------d_logistic_loss-------------------------------------------")
    # real_pred
    # tensor([[0.3418],
    #         [0.7029]], grad_fn= < AddmmBackward0 >)
    # fake_pred
    # tensor([[0.8380],
    #         [0.5778]], grad_fn= < AddmmBackward0 >)

    # real_pred -> torch.Size([2, 1])
    # fake_pred -> torch.Size([2, 1])

    # Большое сходство с La из статьи

    # print("real_pred ->", real_pred.shape)
    # print("fake_pred ->", fake_pred.shape)
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    # print("--------------------------------------------d_r1_loss-------------------------------------------")

    # real_pred -> torch.Size([2, 1])
    # real_img -> torch.Size([2, 3, 64, 64])

    # print("real_pred ->", real_pred.shape)
    # print("real_img ->", real_img.shape)

    # https://arxiv.org/pdf/1801.04406.pdf

    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

# g_nonsaturating_loss(fake_pred, losses, fake_img, real_img, degraded_img)
def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None, correlation_features=None,
                         correlation_consider=True, clip_model=None, clip_preprocess=None, device='cuda'):
    # print("--------------------------------------------g_nonsaturating_loss-------------------------------------------")
    # fake_pred -> torch.Size([2, 1])
    # fake_img -> torch.Size([2, 3, 64, 64])
    # real_img -> torch.Size([2, 3, 64, 64])
    # input_img -> torch.Size([2, 3, 64, 64])
    # print("fake_pred ->", fake_pred.shape)
    # print("fake_img ->", fake_img.shape)
    # print("real_img ->", real_img.shape)
    # print("input_img ->", input_img.shape)

    smooth_l1_loss, id_loss = loss_funcs

    loss = F.softplus(-fake_pred).mean()
    # Lc из статьи
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __ = id_loss(fake_img, real_img, input_img)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Здесь считаем корреляцию

    # Здесь прибавляем ее к лоссу

    # print("--------------------------------------------Подсчет лосса--------------------------------------------")
    # loss(La) = tensor(1.7624, grad_fn= < MeanBackward0 >)
    # loss_l1(Lc) = tensor(0.3152, grad_fn= < SmoothL1LossBackward0 >)
    # loss_id(Lf) = tensor(0.3979, grad_fn= < DivBackward0 >)
    # print("loss (La) =", loss)
    # print("loss_l1 (Lc) =", loss_l1)
    # print("loss_id (Lf) =", loss_id)





    # Добавляем новую функцию потерь основанную на КОРРЕЛЯЦИИ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    loss_correlation = torch.tensor(0.0, device=device)
    if correlation_consider:
        correlation_features = correlation_features
        real_features = get_clip_image_features(real_img.detach().cpu().numpy(), clip_model, clip_preprocess)
        fake_features = get_clip_image_features(fake_img.detach().cpu().numpy(), clip_model, clip_preprocess)

        cos = torch.nn.CosineSimilarity(dim=1)
        corr_between_real_corr = cos(real_features, correlation_features)#[0]
        corr_between_fake_corr = cos(fake_features, correlation_features)#[0]

        loss_correlation = torch.abs(corr_between_real_corr - corr_between_fake_corr)
        # print("loss_correlation ->", loss_correlation, flush=True)
    loss_correlation = loss_correlation.mean()
    print("loss_correlation ->", loss_correlation, flush=True)








    # print("corr_between_real_corr ->", corr_between_real_corr)

    # print("loss_id ->", type(loss_id))
    # print("loss_id ->", loss_id)
    # print("loss_correlation ->", type(loss_correlation))
    # print("loss_correlation ->", loss_correlation, flush=True)

    # Увеличиваем коэффициент перед слагаемым отвечающим за корреляцию
    loss += 1.0*loss_l1 + 1.0*loss_id + 1.0*loss_correlation

    return loss, loss_correlation


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def validation(model, lpips_func, args, device):

    # ВРЕМЕННО УБИРАЮ ВАЛИДАЦИЮ
    # НУЖНО ПОДГОТОВИТЬ ДАННЫЕ ДЛЯ ВАЛИДАЦИИ

    print("--------------------------------------------validation--------------------------------------------")


    lq_files = sorted(glob.glob(os.path.join(args.val_dir, 'lq', '*.*')))
    hq_files = sorted(glob.glob(os.path.join(args.val_dir, 'hq', '*.*')))

    print("lq_files ->", lq_files)
    print("hq_files ->", hq_files)


    assert len(lq_files) == len(hq_files)

    dist_sum = 0
    model.eval()
    for lq_f, hq_f in zip(lq_files, hq_files):
        img_lq = cv2.imread(lq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_lq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_t = torch.flip(img_t, [1])
        
        with torch.no_grad():
            img_out, __ = model(img_t)
        
            img_hq = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq = F.interpolate(img_hq, (args.size, args.size))
            dist_sum += lpips_func.forward(img_out, img_hq)
    
    return dist_sum.data/len(lq_files)

def get_clip_image_features(correlation_img, clip_model, clip_preprocess):
    # Обратные проеобразования для правильной формы входа в CLIP
    correlation_img_copy = correlation_img
    correlation_img_copy = (correlation_img_copy * 0.5) + 0.5
    correlation_img_copy = (correlation_img_copy * 255.).astype(np.uint8)
    correlation_img_copy = correlation_img_copy.transpose(0, 2, 3, 1)

    with torch.no_grad():
        # clip_preprocess не поддерживает батчинг
        clip_image_inputs = []
        for c_img in correlation_img_copy:
            pil_img = Image.fromarray(c_img)

            # Prepare the inputs
            if len(clip_image_inputs) == 0:
                clip_image_inputs = clip_preprocess(pil_img).unsqueeze(0)
            else:
                clip_image_inputs = torch.cat((clip_image_inputs, clip_preprocess(pil_img).unsqueeze(0)))

        clip_image_inputs = clip_image_inputs.to(device)

        # Calculate features
        clip_image_features = clip_model.encode_image(clip_image_inputs)
        # print(clip_image_features.shape)
        # print(type(clip_image_features))
        # print(clip_image_features)
        return clip_image_features.to(torch.float32)

def train(args, loader, generator, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, clip_model, clip_preprocess, device):
    print("--------------------------------------------TRAIN-------------------------------------------", flush=True)
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))

    loader = sample_data(loader)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    ##############################################################################################################
    d_loss_val = 0
    ##############################################################################################################
    r1_loss = torch.tensor(0.0, device=device)
    ##############################################################################################################
    g_loss_val = 0
    ##############################################################################################################
    path_loss = torch.tensor(0.0, device=device)
    ##############################################################################################################
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
 
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        # [4, 3, 256, 256] - сейчас
        # [4, 2, 3, 256, 256] - нужно теперь
        degraded_img, real_img, correlation_img = next(loader)


        # В место коррелируемого кадра мы теперь хотим отправлять его признаки подсчитаныне с помощью CLIP
        correlation_features = get_clip_image_features(correlation_img.detach().cpu().numpy(), clip_model, clip_preprocess)




        # Вероятность получить черное изображение в место коррелируемого (На первый раз пробуем 50% шанс)
        zero_correlation_chance = 0.05
        # Учитываем ли корреляцию при подсчете лосса генератора
        correlation_consider = True  # По умолчанию
        # Если сработает то заменяем корреляционное изображние - пустым, и не учитываем при подсчете лосса генератора
        if np.random.uniform() < zero_correlation_chance:
            correlation_img = torch.zeros_like(correlation_img)
            correlation_features = torch.zeros_like(correlation_features)
            correlation_consider = False
            print("Пустое корреляционное изображение", flush=True)
        else:
            print("Не пустое корреляционное изображение", flush=True)

        correlation_img = correlation_img.to(device)
        correlation_features = correlation_features.to(device)
        degraded_img = degraded_img.to(device)
        real_img = real_img.to(device)

        # Сначала градиент касается только дискриминатора
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # Второе None
        # На всход генератор должен также получать коррелируемое изображение
        fake_img, _ = generator(degraded_img, correlation_features)





        # print("generator (fake_img) ->", fake_img.shape)
        fake_pred = discriminator(fake_img)
        # print("discriminator (fake_pred) ->", fake_pred.shape)

        real_pred = discriminator(real_img)
        # print("discriminator (real_pred) ->", type(real_pred))

        ##############################################################################################################
        # loss дискриминатора
        # Предпологаю это La (теперь уверен)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        # print("d_logistic_loss", d_loss)

        ##############################################################################################################
        # Значение функции ошибки дискриминатора
        loss_dict['d'] = d_loss
        # Среднее значение предикта дискриминатора на реальном изображении
        loss_dict['real_score'] = real_pred.mean()
        # Среднее значение предикта дискриминатора на сгенерированном изображении
        loss_dict['fake_score'] = fake_pred.mean()

        # Обратное распространение ошибки для дискриминатора
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Периодичность регуляризации дискриминатора
        d_regularize = i % args.d_reg_every == 0

        # Регуляризация каждые 16 итераций
        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            ##############################################################################################################
            # R1-регуляризация не описана в статье
            r1_loss = d_r1_loss(real_pred, real_img)

            # https://arxiv.org/pdf/1801.04406.pdf

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        ##############################################################################################################
        # loss r1 регуляризации дискриминатора
        loss_dict['r1'] = r1_loss

        # Теперь градиент касается только генератора
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # Добавляем корреляционный вход
        fake_img, _ = generator(degraded_img, correlation_features)
        # print("generator (fake_img) ->", fake_img.shape)
        fake_pred = discriminator(fake_img)
        # print("discriminator (fake_pred) ->", fake_pred.shape)

        ##############################################################################################################
        # loss генератора
        # print("In", fake_pred.shape, fake_img.shape, real_img.shape, degraded_img.shape)
        # Добавляем в лосс генератора КОРРЕЛЯЦИЮ
        # Добавляем корреляционный вход
        g_loss, loss_correlation = g_nonsaturating_loss(
            fake_pred, losses, fake_img, real_img, degraded_img,
            correlation_features=correlation_features, correlation_consider=correlation_consider,
            clip_model=clip_model, clip_preprocess=clip_preprocess, device=device)
        ##############################################################################################################
        loss_dict['g'] = g_loss
        loss_dict['corr'] = loss_correlation

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)


            # Добавляем корреляционный вход
            fake_img, latents = generator(degraded_img, correlation_features, return_latents=True)
            ##############################################################################################################
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            ##############################################################################################################
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        ##############################################################################################################
        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        ##############################################################################################################
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        corr_loss_val = loss_reduced['corr'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        # print("path_loss_val =", path_loss_val)
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        # print("get_rank() ->", get_rank())
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; corr: {corr_loss_val:.4f}'
                )
            )

            if i % (args.save_freq // 100) == 0:

                with torch.no_grad():
                    g_ema.eval()

                    # Добавляем корреляционный вход
                    sample, _ = g_ema(degraded_img, correlation_features)
                    # print("SAMPLE 1", sample.shape)
                    sample = torch.cat((correlation_img, degraded_img, sample, real_img), 0)
                    # print("SAMPLE 2", sample.shape)
                    utils.save_image(
                        sample,
                        f'{args.sample}/{str(i).zfill(6)}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )

                # ВРЕМЕННО УБИРАЮ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # lpips_value = validation(g_ema, lpips_func, args, device)
                # print(f'{i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]}')
                # print("NOT VALIDATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if i and i % args.save_freq == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'{args.ckpt}/{str(i).zfill(6)}.pth',
                )



# CUDA_VISIBLE_DEVICES='0,1,2,3'
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321
# train_simple.py
# --size 1024
# --channel_multiplier 2
# --narrow 1
# --ckpt weights
# --sample results
# --batch 2
# --path your_path_of_croped+aligned_hq_faces (e.g., FFHQ)
# ваш путь обрезанных+выровненных лиц hq (например, FFHQ)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Желательно знать для чего нужен каждый аргумент
    print("PARSER", flush=True)
    # path your_path_of_croped+aligned_hq_faces
    parser.add_argument('--path', type=str, default='examples/test_dataset')
    parser.add_argument('--base_dir', type=str, default='./')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--iter', type=int, default=600000)
    # parser.add_argument('--iter', type=int, default=4000000)





    # batch 2
    # parser.add_argument('--batch', type=int, default=2)
    # parser.add_argument('--batch', type=int, default=4)

    parser.add_argument('--batch', type=int, default=8)






    # size 1024
    # Из-за id_loss минимальный размер по задумке 256
    # Из-за нехватки памяти вынужден переделать id_loss и поставить 64
    parser.add_argument('--size', type=int, default=512)
    # parser.add_argument('--size', type=int, default=256)
    # parser.add_argument('--size', type=int, default=64)







    # channel_multiplier 2
    # Из-за нехватки памяти 1
    parser.add_argument('--channel_multiplier', type=int, default=2)
    # parser.add_argument('--channel_multiplier', type=int, default=1)







    # narrow 1
    parser.add_argument('--narrow', type=float, default=1.0)

    parser.add_argument('--r1', type=float, default=10)

    parser.add_argument('--path_regularize', type=float, default=2)
    # Для регуляризации генератора
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    # На какой итерации производится регуляризации дискриминатора
    parser.add_argument('--d_reg_every', type=int, default=16)
    # Параметр для оптимизатора / На какой итерации производится регуляризации генератора
    parser.add_argument('--g_reg_every', type=int, default=4)
    # Логирование в виде изображений и сохранение весов раз в 10000 итераций
    parser.add_argument('--save_freq', type=int, default=5000)
    # parser.add_argument('--save_freq', type=int, default=25000)
    # Параметр для оптимизатора
    parser.add_argument('--lr', type=float, default=0.002)

    parser.add_argument('--local_rank', type=int, default=0)
    # ckpt weights
    parser.add_argument('--ckpt', type=str, default='ckpts-2')









    # parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default='ckpts-2/040000.pth')
    # parser.add_argument('--pretrain', type=str, default=None)







    # sample results
    parser.add_argument('--sample', type=str, default='sample')

    parser.add_argument('--val_dir', type=str, default='val')

    # Аргументы достаем из этой переменной
    args = parser.parse_args()

    # Выяснить для чего создаются эти директории

    # default='ckpts'
    # Папка для бекапа весов
    os.makedirs(args.ckpt, exist_ok=True)
    # default = 'sample'
    # Папка для логирования работы сети в виде изображений
    os.makedirs(args.sample, exist_ok=True)

    # Попробовать использовать 'cpu' в место 'cuda'
    device = 'cuda'
    # device = 'cpu'

    # Скорее всего имеется в виду количество видеокарт
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    # На случай если у нас больше одной видеокарты
    # У нас всегда одна видеокарта
    # Можно будет закомментировать
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # Не ясно для чего
    args.latent = 512
    # number of multi-layer perception layers
    # количество слоев многослойного восприятия
    args.n_mlp = 8







    # Стартовая итерация
    # Видимо, для пауз процесса обучения
    # args.start_iter = 370001
    args.start_iter = 40001








    # Разобрать !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Модель генератора
    # ('--size', type=int, default=256)
    # args.latent = 512
    # args.n_mlp = 8
    # ('--channel_multiplier', type=int, default=2)
    # ('--narrow', type=float, default=1.0)
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    generator = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    # # Разобрать !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # Дискриминатор
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )


    # Если модель обучалась ранее
    if args.pretrain is not None:
        print('load model:', args.pretrain, flush=True)

        ckpt = torch.load(args.pretrain)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    id_loss = IDLoss(args.base_dir, device, ckpt_dict=None)

    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)
    #
    # if args.distributed:
    #     generator = nn.parallel.DistributedDataParallel(
    #         generator,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )
    #
    #     discriminator = nn.parallel.DistributedDataParallel(
    #         discriminator,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )
    #
    #     id_loss = nn.parallel.DistributedDataParallel(
    #         id_loss,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )
    #
    dataset = FaceDataset(args.path, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    clip_model, clip_preprocess = clip.load('RN50', device)

    train(args, loader, generator, discriminator, [smooth_l1_loss, id_loss], g_optim, d_optim, g_ema, lpips_func, clip_model, clip_preprocess, device)
    # def train(args, loader, generator, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, device):