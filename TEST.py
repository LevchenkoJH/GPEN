#########################################################################################################################
# # Для подготовки датасета
# import os
# import glob
#
#
# # path = '/home/jasmine/VGGFace2-HQ/Data/VGGface2_FFHQalign'
# path = '/home/levchenkone/GITHUB/GPEN/examples/test_dataset'
#
# HQ_imgs = glob.glob(os.path.join(path, '*', '*'))
#
# # print(len(HQ_imgs))
#
#
# count = 0
# for p in HQ_imgs:
#     images_path = glob.glob(os.path.join(p, '*.*'))
#     # print(len(images_path))
#     if (len(images_path) == 0):
#         print(p)
#         os.rmdir(p)
#         count +=1
#     if (len(images_path) == 1):
#         print(p)
#         os.remove(images_path[0])
#         count += 1
#
# print("count =", count)
#########################################################################################################################
# # Хотим переместить датасет
# import os
# import glob
#
# # path = '/home/jasmine/VGGFace2-HQ/Data/VGGface2_FFHQalign'
# path = r'D:\GPEN\GPEN\examples\test_dataset'
#
# # HQ_imgs = glob.glob(os.path.join(path, '*', '*coef*'))
# HQ_imgs = glob.glob(os.path.join(path, '*', '*'))
#
#
# print(HQ_imgs)
# for name in HQ_imgs:
#     print(name)
#     # os.rename(name, name[0:-12])
#########################################################################################################################
# import numpy as np
#
# for i in range(10):
#     print(np.random.uniform(), np.random.uniform())
#########################################################################################################################
import torch.nn.functional as F
import torch
source = torch.rand((2,2, 2))
# now we expand to size (7, 11) by appending a row of 0s at pos 0 and pos 6,
# and a column of 0s at pos 10
result = F.pad(input=source, pad=(2, 2, 2, 2), mode='constant', value=0)
print(result.shape)
print(result)
