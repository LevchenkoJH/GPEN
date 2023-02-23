import torch
torch.cuda.empty_cache()


import os
import glob


path = '/home/jasmine/VGGFace2-HQ/Data/VGGface2_FFHQalign'

HQ_imgs = glob.glob(os.path.join(path, '*', '*coef*'))

# print(len(HQ_imgs))


count = 0
for p in HQ_imgs:
    images_path = glob.glob(os.path.join(p, '*.*'))
    # print(len(images_path))
    if (len(images_path) == 0):
        print(p)
        os.rmdir(p)
        count +=1
    if (len(images_path) == 1):
        print(p)
        os.remove(images_path[0])
        count += 1

print("count =", count)