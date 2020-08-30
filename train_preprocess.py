############################################################
# This file is for to preprocess images from Planet Lab    #
# to convert them from 4 bands to 3 bands and crop them to #
# a fixed size                                             #
############################################################

import json
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict 

print('Start Processing')
def process_mask(masks_dir, masks_out):
    masks_filenames = [f for f in os.listdir(masks_dir)]
    for mask in masks_filenames:
        image_id = mask.split('.')[0] #get image ID
        im_name = image_id + '.png'
        mask_image = Image.open(masks_dir + im_name)
        mask_image.load()
        background = mask_image.crop((0, 0, cut_size, cut_size)) 
        # save mask
        overlay_path = masks_out + image_id + '.png'
        background.save(overlay_path, 'png', quality=100)
        
def process_image(imgs_dir, imgs_out):
    imgs_filenames = [f for f in os.listdir(imgs_dir)]
    for img_file in imgs_filenames:
        image_id = img_file.split('.')[0] #get image ID
        im_name = image_id + '.tif'
        rgba_image = Image.open(imgs_dir + im_name)
        #turn to RGB
        rgba_image.load()
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask = rgba_image.split()[3])
        #crop to a unanimous size
        background = background.crop((0, 0, cut_size, cut_size)) 
        #save image
        overlay_path = imgs_out + image_id + '.jpeg'
        background.save(overlay_path, 'JPEG', quality=100)


imgs_dir = 'data/imgs_planet/'
masks_dir = 'data/masks_planet/'
imgs_out = 'data/imgs/'
masks_out = 'data/masks/'
cut_size = 467

# comment out the function if don't want to process the images or the masks 
process_image(imgs_dir, imgs_out)
process_mask(masks_dir, masks_out)
           
