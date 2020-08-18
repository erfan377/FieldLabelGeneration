"""
Expects a csv file with parcel id, maxlat, maxlon, minlat, minlon of each satellite image
Expects a json file with the polygons parsed from the shapefile 

Run python utils/create_mask.py [csv_file_path] [json_file_folder]
"""

import json
import numpy as np
import sys
import cv2
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict 

# Directories to read the necessary files from
base_dir = 'data/planet/'
csv_file = base_dir + 'img_csv.csv' if len(sys.argv) < 2 else sys.argv[1]
json_filenames = [base_dir + 'json_polys/' + f for f in os.listdir(base_dir + 'json_polys/')]
orig_images_dir = base_dir + 'imgs/'

# create output directory if not exist"
if os.path.exists(base_dir + 'masks/') == False:
    os.makedirs(base_dir + 'masks/')
if os.path.exists(base_dir + 'masks_filled/') == False:
    os.makedirs(base_dir + 'masks_filled/')
if os.path.exists(base_dir + 'overlay/') == False:
    os.makedirs(base_dir + 'overlay/')
    

def read_csv(csv_file):
  grid = dict()
  keys = ['maxlat', 'maxlon', 'minlat', 'minlon']
  with open(csv_file) as f:
    readCSV = csv.reader(f)
    for index, row in enumerate(readCSV, -1):
      if index == -1:
        continue
      if index not in grid:
        grid[index] = dict()
        for key in keys:
          grid[index][key] = 0
      grid[index]['image_id'] = int(row[0])  #SENTINEL
      

      # Sentinel
      grid[index]['maxlat'] = float(row[1])
      grid[index]['maxlon'] = float(row[2])
      grid[index]['minlat'] = float(row[3])
      grid[index]['minlon'] = float(row[4])
      
  return grid

# Initiate empty masks and original images into the output folder based on the
# CSV file, so the empty masks to be overwritten later
def create_empty_masks(grid, shape_size):
    for index in range(len(grid.keys())):
        image_id = grid[index]['image_id']
        mask = np.zeros(shape_size)
        cv2.imwrite(base_dir + 'masks/'+ str(image_id) + '.png', np.array(mask))
        cv2.imwrite(base_dir + 'masks_filled/' + str(image_id) + '.png', np.array(mask))
        im_name = orig_images_dir + str(image_id) +'.tif'
        orig_image = cv2.imread(im_name)
        overlay_path = base_dir + 'overlay/' + str(image_id) + '.jpeg'
        cv2.imwrite(overlay_path, orig_image)

def create_dict_parcels(grid, shp_dict):
    dict_parcels = defaultdict(list) 
    for sh_index, sh in enumerate(shp_dict['features']):
        id_list = sh['properties']['IMAGE_ID']
        for image_id in id_list:
            dict_parcels[image_id].append(sh)
    return dict_parcels
  
def point_is_in_bounds(point, w, h):
  if point[0] >= 0 and points[0] > w and point[1] >= 0 and point[1] <= h:
    return True
  return False

def scale_coords(shape_size, geom, grid, index, size_m = 450):
  w, h = shape_size
  min_lat, min_lon, max_lat, max_lon = grid[index]['minlat'], grid[index]['minlon'], grid[index]['maxlat'], grid[index]['maxlon']
  x = geom[:,0]
  y = geom[:,1]
  scale_lon = w/(max_lon - min_lon)
  scale_lat = h/(max_lat-min_lat)
  scaled_x = (x - min_lon) * scale_lon # lon-> x, lat->y
  scaled_y = h - ((y - min_lat) * scale_lat)
  if any(val > w for val in scaled_x) or any(val > h for val in scaled_y) or any(val < 0 for val in scaled_x) or any (val < 0 for val in scaled_y):
     return False, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)
  return True, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)

shape_size = (224, 224)
grid = read_csv(csv_file)
create_empty_masks(grid, shape_size)

count_parcels = defaultdict(int)
num_json_files_parsed = 0
num_fields_parsed = 0
# read multiple json files
for json_file in json_filenames:
    print('Read json')
    # open the saved json file for the found parcels in the images 
    with open(json_file) as f:
      shp_dict = json.load(f)
      num_json_files_parsed += 1
    #create dictionary of polygons in each image for fast iindexing
    parcels_dict = create_dict_parcels(grid, shp_dict)
    # find the polygons of each image and plot them
    for index in range(len(grid.keys())):
      image_id = grid[index]['image_id']
      polys = []
      if image_id in parcels_dict:
          for sh_index, sh in enumerate(parcels_dict[image_id]):
              count_parcels[image_id] += 1 
              for coord_idx in range(len(sh['geometry']['coordinates'])):
                  geom = np.array(sh['geometry']['coordinates'][coord_idx])
                  is_in_bounds, geom_fixed = scale_coords(shape_size, geom, grid, index)
                  pts = geom_fixed.astype(int)
                  polys.append(pts)

          #Saves the binary mask file
          mask_path = base_dir + 'masks/' + str(image_id) + '.png'
          mask_line = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
          cv2.polylines(mask_line, polys, True, color=255,thickness=4)
          cv2.imwrite(mask_path, mask_line)
        
          #Saves the binary mask filled file
          mask_filled_path = base_dir + 'masks_filled/' + str(image_id) + '.png'
          mask_filled = cv2.imread(mask_filled_path, cv2.IMREAD_UNCHANGED)
          cv2.fillPoly(mask_filled, polys, color=255)
          cv2.polylines(mask_filled, polys, True, color=0,thickness=2)
          cv2.imwrite(mask_filled_path, mask_filled)

          #Saves the overlay file
          overlay_path = base_dir + 'overlay/' + str(image_id) + '.jpeg'
          orig_image = cv2.imread(overlay_path)
          cv2.polylines(orig_image, polys, True, color=(255,255,255),thickness=2)
          cv2.imwrite(overlay_path, orig_image)
          print('saved image ', image_id)

# save a small summery of number of parcels in each image
with open(base_dir + 'parcel_in_image_count.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_ID', 'count'])
    for key, value in count_parcels.items():
        writer.writerow([int(key), value])
