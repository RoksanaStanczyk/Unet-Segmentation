from PIL import Image
import glob
import os
import cv2
import numpy as np

image_list = []
for filename in glob.glob(r'D:\Unet_project\nucleus_images_blue\images\*'):
    image=Image.open(filename)
    image = resize.prepare(image)
    image_list.append(image)

# from matplotlib import pyplot as plt
# plt.imshow(image_list[::10], interpolation='nearest')
# plt.show()