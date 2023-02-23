#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import kendalltau
import time



# In[2]:


def load_image(path, title='', show=False):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if show:
        show_image(image, path)
    return image


# In[3]:


def show_image(image, title='', size=(15, 15)):
    plt.figure(figsize = size)
    if title != '':
        plt.title(title)
    plt.imshow(image)
    plt.show()
    plt.clf()


# # Коэффициент Танимото

# In[4]:


# От -1 до 1
def tanimoto_coefficient(image1, image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")
    
    S1 = np.sum(image1 * image2)
    S2 = np.sum((image1 - image2) ** 2)
    St = S1 / (S1 + S2)
    if prin:
        print("Коэффициент Танимото ->", St)
    return St


# # Кросс-корреляция изображений

# In[5]:


# От -1 до 1
def cross_correlation_coefficient(image1, image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")
    
    # Объем локальной выборки
    h, w = image1.shape
    m = h * w
#     print("m ->", m)
    # Оценкка математического ожидания изображения
    M1 = np.sum(image1) / m
    M2 = np.sum(image2) / m
#     print("M1 ->", M1)
#     print("M2 ->", M2)
    # Оценка дисперсии изображения
    D1 = np.sqrt(np.sum((image1 - M1) ** 2) / m)
    D2 = np.sqrt(np.sum((image2 - M2) ** 2) / m)
#     print("D1 ->", D1)
#     print("D2 ->", D2)
    
    r = np.sum((image1 - M1) * (image2 - M2)) / (m * D1 * D2)
    
    if prin:
        print("Кросс-корреляция ->", r)
    return r


# # Коэффициент ранговой корреляции Кендалла

# In[6]:


def kendall_coefficient(torch_image1, torch_image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    # так и оказалось

    # torch_image1 -> torch.Size([2, 3, 64, 64])
    # torch_image2 -> torch.Size([2, 3, 64, 64])
    # print("torch_image1 ->", torch_image1.shape)
    # print("torch_image2 ->", torch_image2.shape)

    # Переводим в numpy-массив
    image1 = torch_image1.detach().cpu().numpy()
    image2 = torch_image2.detach().cpu().numpy()

    # image1 numpy -> (2, 3, 64, 64)
    # image2 numpy -> (2, 3, 64, 64)
    # print("image1 numpy ->", image1.shape)
    # print("image2 numpy ->", image2.shape)


    # Производим обратные действия относительно dataset_face __get+item__
    image1 = (image1 * 0.5) + 0.5
    image2 = (image2 * 0.5) + 0.5

    image1 = (image1 * 255.).astype(np.uint8)
    image2 = (image2 * 255.).astype(np.uint8)

    image1 = np.reshape(image1, (-1, image1.shape[2], image1.shape[3], image1.shape[1]))
    image2 = np.reshape(image2, (-1, image2.shape[2], image2.shape[3], image2.shape[1]))

    # image1 -> (2, 64, 64, 3)
    # image2 -> (2, 64, 64, 3)
    # print("image1 ->", image1.shape)
    # print("image2 ->", image2.shape)

    # Даталоадер подгружает изображения в RGB
    # cvtColor не работает для батчей
    gray_image1 = []
    gray_image2 = []
    for i in range(len(image1)):
        # print("Изображение", i)
        if len(gray_image1) == 0:
            gray_image1 = np.array([cv2.cvtColor(image1[i], cv2.COLOR_RGB2GRAY)])
        else:
            gray_image1 = np.append(gray_image1, [cv2.cvtColor(image1[i], cv2.COLOR_RGB2GRAY)], axis=0)
        # print("gray_image1 ->", gray_image1.shape)

    for i in range(len(image2)):
        # print("Изображение", i)
        if len(gray_image2) == 0:
            gray_image2 = np.array([cv2.cvtColor(image2[i], cv2.COLOR_RGB2GRAY)])
        else:
            gray_image2 = np.append(gray_image2, [cv2.cvtColor(image2[i], cv2.COLOR_RGB2GRAY)], axis=0)
        # print("gray_image2 ->", gray_image2.shape)

    image1 = gray_image1
    image2 = gray_image2

    # image1 -> <class 'numpy.ndarray'>
    # image2 -> <class 'numpy.ndarray'>
    # print("image1 ->", type(image1))
    # print("image2 ->", type(image2))

    # image1 -> (2, 64, 64)
    # image2 -> (2, 64, 64)
    # print("image1 ->", image1.shape)
    # print("image2 ->", image2.shape)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")

    image1 = image1.reshape(-1, image1.shape[1]*image1.shape[2])
    image2 = image2.reshape(-1, image2.shape[1]*image2.shape[2])

    # image1 -> (2, 4096)
    # image2 -> (2, 4096)
    # print("image1 ->", image1.shape)
    # print("image2 ->", image2.shape)

    # Считаем корреляцию не между батчами,
    # а среднюю корреляцию между изображениями батчей
    batch_coef = np.array([])
    for i in range(len(image1)):
        coef, _ = kendalltau(image1[i], image2[i])
        batch_coef = np.append(batch_coef, coef)
    # print("batch_coef ->", batch_coef.mean())
    
    if prin:
        print('Коэффициент ранговой корреляции Кендалла ->', batch_coef.mean())
    
    return batch_coef.mean()


# # Одна пара изображений

# In[7]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/input"
#
#     file_names = sorted(os.listdir(IN_PATH))
#
#     print(file_names)
#
#     image_path_1 = os.path.join(IN_PATH, file_names[0])
#     image_path_2 = os.path.join(IN_PATH, file_names[2])
#     print(image_path_1)
#     print(image_path_2)
#
#
#     image_1 = load_image(image_path_1)
#     image_2 = load_image(image_path_2)
#
#     print('\n')
#     time_start = time.time()
#     buf1 = tanimoto_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
#     time_start = time.time()
#     buf2 = cross_correlation_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
#     time_start = time.time()
#     buf3 = kendall_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
# main()


# # Множество пар изображений

# In[8]:


# def many_test(IN_PATH, file_names):
#     previous_file = ""
#
#     points1 = np.array([0])
#     points2 = np.array([0])
#     points3 = np.array([0])
#
#     for file in file_names:
#         if (previous_file != ""):
# #             print(previous_file + " " + file)
#             image1 = load_image(path=os.path.join(IN_PATH, previous_file), show=False)
#             image2 = load_image(path=os.path.join(IN_PATH, file), show=False)
#
#             buf1 = tanimoto_coefficient(image1, image2, show=False)
#             buf2 = cross_correlation_coefficient(image1, image2)
#             buf3 = kendall_coefficient(image1, image2)
#
#             points1 = np.append(points1, buf1)
#             points2 = np.append(points2, buf2)
#             points3 = np.append(points3, buf3)
#
#
#
#         previous_file = file
#
#
#     plt.figure(figsize = (15, 5))
#     plt.axis([0, 228, 0.6, 1.0])
#     plt.plot(range(len(file_names)), points1, 'r')
#     plt.plot(range(len(file_names)), points2, 'g')
#     plt.plot(range(len(file_names)), points3, 'b')
#     plt.legend(['Коэффициент Танимото',
#                 'Кросс-корреляция изображений',
#                 'Коэффициент ранговой корреляции Кендалла'], loc=0)
#     plt.show()


# In[9]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/_CuZqXrhEZI_5.mp4222"
#     file_names = sorted(os.listdir(IN_PATH))
#     many_test(IN_PATH, file_names)
# main()


# In[10]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/_CuZqXrhEZI_5.mp4"
#     file_names = sorted(os.listdir(IN_PATH))
#     many_test(IN_PATH, file_names)
# main()


# In[ ]:




