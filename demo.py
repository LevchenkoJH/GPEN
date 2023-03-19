'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import __init_paths
from face_enhancement import FaceEnhancement
from face_colorization import FaceColorization
from face_inpainting import FaceInpainting
from segmentation2face import Segmentation2Face
from tqdm import tqdm

def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

def make_mp4_video(input_dir, output_dir, fps, video_name):
    # dirs = sorted(os.listdir(input_dir))
    #
    # for frame_path in tqdm(dirs):

    # frame_path_tmp = os.path.join(input_dir, frame_path)
    # print(frame_path_tmp)
    img_array = []
    files = sorted(glob.glob(input_dir + '/*.png'))
    # print(files)
    size = (0, 0)
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out = cv2.VideoWriter(os.path.join(output_dir, video_name) + '.mp4',
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__=='__main__':

    # Выбираем параметры

    parser = argparse.ArgumentParser()
    # Можно указать свою модель
    # Либо вообще убрать этот аргумент
    parser.add_argument('--model', type=str, default='170000', help='GPEN model')
    # parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    # parser.add_argument('--task', type=str, default='FaceEnhancement', help='task of GPEN model')
    parser.add_argument('--key', type=str, default=None, help='key of GPEN model')
    # Мы обучали для 256
    # parser.add_argument('--in_size', type=int, default=512, help='in resolution of GPEN')
    # parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')

    parser.add_argument('--in_size', type=int, default=256, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=256, help='out resolution of GPEN')

    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--alpha', type=float, default=1, help='blending the results')
    #
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    #
    parser.add_argument('--use_cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--save_face', action='store_true', help='save face or not')
    parser.add_argument('--aligned', action='store_true', help='input are aligned faces or not')
    parser.add_argument('--sr_model', type=str, default='realesrnet', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--tile_size', type=int, default=0, help='tile size for SR to avoid OOM')
    parser.add_argument('--indir', type=str, default='examples/video', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-BFR', help='output folder')
    parser.add_argument('--ext', type=str, default='.jpg', help='extension of output')
    parser.add_argument('--buf_dir', type=str, default='buffer', help='Папка для помежуточных изображений')
    parser.add_argument('--buf_frames_dir', type=str, default='frames', help='Папка с необработанными кадрами')
    parser.add_argument('--buf_gpen_frames_dir', type=str, default='gpen', help='Папка с обработанными кадрами')
    parser.add_argument('--buf_gpen_frames_dir_prod', type=str, default='gpen_prod', help='Папка с обработанными кадрами')
    args = parser.parse_args()

    #model = {'name':'GPEN-BFR-512', 'size':512, 'channel_multiplier':2, 'narrow':1}
    #model = {'name':'GPEN-BFR-256', 'size':256, 'channel_multiplier':1, 'narrow':0.5}

    # Расположение результата работы
    os.makedirs(args.outdir, exist_ok=True)
    # У нас только FaceEnhancement
    processer = FaceEnhancement(args, in_size=args.in_size, out_size=args.out_size, model=args.model, use_sr=args.use_sr, device='cuda' if args.use_cuda else 'cpu')
    # Обрабатываемые видео
    dirs = sorted(os.listdir(args.indir))
    print(dirs)

    for video_path in dirs:
        print(">>>>>>>>>>>>>>>> Обработка", video_path)
        # Расшариваем видео в буфере (frames)
        video_path_tmp = os.path.join(args.indir, video_path)
        videoCapture = cv2.VideoCapture()
        videoCapture.open(video_path_tmp)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Буфер изображений
        print(">>>>>>>> Считывание кадров <<<<<<<<")
        # Папка для помежуточных изображений
        buf_dir = args.buf_dir
        # Папка с необработанными кадрами
        buf_frames_dir = args.buf_frames_dir
        # Папка с обработанными кадрами
        buf_gpen_frames_dir = args.buf_gpen_frames_dir

        for i in tqdm(range(int(frames))):
            ret, frame = videoCapture.read()
            if not os.path.exists(buf_dir):
                os.mkdir(buf_dir)
            if not os.path.exists(os.path.join(buf_dir, video_path)):
                os.mkdir(os.path.join(buf_dir, video_path))
            if not os.path.exists(os.path.join(buf_dir, video_path, buf_frames_dir)):
                os.mkdir(os.path.join(buf_dir, video_path, buf_frames_dir))
            # Сохраняем кадры на диске
            cv2.imwrite(os.path.join(buf_dir, video_path, buf_frames_dir, str(i).zfill(6)) + ".png", frame)

        # Обрабатываем изображения из буффера
        print(">>>>>>>> Обработка видео <<<<<<<<")
        buf_frames_dir_list = sorted(os.listdir(os.path.join(buf_dir, video_path, buf_frames_dir)))
        for i in tqdm(range(len(buf_frames_dir_list))):
            file_name = os.path.join(buf_dir, video_path, buf_frames_dir, buf_frames_dir_list[i])
            img = cv2.imread(file_name, cv2.IMREAD_COLOR) # BGR

            if i == 0:
                corr_img = np.zeros_like(img) # BGR
            else:
                corr_img = cv2.imread(os.path.join(buf_dir, video_path, buf_frames_dir, buf_frames_dir_list[i - 1]), cv2.IMREAD_COLOR) # BGR

            if not isinstance(img, np.ndarray) or not isinstance(corr_img, np.ndarray):
                print("ids:", i - 1, i, 'error')
                continue

            img_out, orig_faces, enhanced_faces = processer.process(img=img, corr_img=corr_img, isFirst=i == 0, aligned=args.aligned)
            # Для сравнения
            img = cv2.resize(img, img_out.shape[:2][::-1])

            # Папка для помежуточных изображений
            buf_dir = args.buf_dir
            # Папка с обработанными кадрами
            buf_gpen_frames_dir = args.buf_gpen_frames_dir
            if not os.path.exists(os.path.join(buf_dir, video_path, buf_gpen_frames_dir)):
                os.mkdir(os.path.join(buf_dir, video_path, buf_gpen_frames_dir))

            # Записываем оригинальный и обработанный кадр в одном видео
            cv2.imwrite(os.path.join(buf_dir, video_path, buf_gpen_frames_dir, str(i).zfill(6) + ".png"), np.hstack((img, img_out)))

            # Папка с обработанными кадрами
            buf_gpen_frames_dir_prod = args.buf_gpen_frames_dir_prod
            if not os.path.exists(os.path.join(buf_dir, video_path, buf_gpen_frames_dir_prod)):
                os.mkdir(os.path.join(buf_dir, video_path, buf_gpen_frames_dir_prod))
            # Записываем только обработанный кадр
            cv2.imwrite(os.path.join(buf_dir, video_path, buf_gpen_frames_dir_prod, str(i).zfill(6) + ".png"), img_out)

            # Для сохранения обработанных лиц
            if args.save_face:
                for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                    of = cv2.resize(of, ef.shape[:2])
                    cv2.imwrite(os.path.join(args.outdir, '.'.join((video_path + str(i)).split('.')[:-1])+'_face%02d'%m+args.ext), np.hstack((of, ef)))

        print(">>>>>>>> Сбор видео <<<<<<<<")
        # Нужно собрать 2 видео
        # Сбор видео
        frames_buf_dirs = sorted(os.listdir(os.path.join(buf_dir, video_path, buf_gpen_frames_dir)))
        make_mp4_video(input_dir=os.path.join(buf_dir, video_path, buf_gpen_frames_dir), output_dir=args.outdir, fps=fps, video_name=os.path.splitext(video_path)[0] + '+Сompare')
        make_mp4_video(input_dir=os.path.join(buf_dir, video_path, buf_gpen_frames_dir_prod), output_dir=args.outdir, fps=fps, video_name=os.path.splitext(video_path)[0])