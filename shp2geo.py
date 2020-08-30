import shapefile
from json import dumps
import fiona
from pyproj import Proj#, transform
import pyproj
import numpy as np
import random
import csv
import shapely
from shapely.geometry import Polygon
from shapely.geometry import shape
from functools import partial
from shapely.ops import transform
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from collections import defaultdict


def read_csv(shape_file, csv_file='./data/img_bbox.csv'):
  """Read the coordinate of the bounding boxes and constructs and R-Tree data structure

  Args:
      shape_file : polygons
      csv_file (str, optional): csv containing bounding boxes. The path is usually given before.

  Returns:
    dict, r-tree: dict of bounding boxes for each image id and r-tree
  """
  original = Proj(fiona.open(shape_file).crs)
  destination = Proj('epsg:4326')

  grid = dict()
  keys = ['maxlat', 'maxlon', 'minlat', 'minlon']
  poly_list = []
  with open(csv_file) as f:
    readCSV = csv.reader(f)
    for index, row in enumerate(readCSV, -1):
      if index == -1:
        continue
      if index not in grid:
        grid[index] = dict()
        for key in keys:
          grid[index][key] = 0
      grid[index]['Image_id'] = int(row[0])
      
      maxlat = float(row[1])
      maxlon = float(row[2])
      minlat = float(row[3])
      minlon = float(row[4])

      grid[index]['poly'] = shapely.geometry.box(minlat, minlon, maxlat, maxlon) #Polygon([(minlon, minlat), (minlon, maxlat), (maxlon, maxlat),(maxlon, minlat)])
      project = partial(pyproj.transform, original, destination)
      grid[index]['poly'] = transform(project, grid[index]['poly'])
      
      #populating r-tree
      poly_obj = grid[index]['poly']
      poly_obj.name = grid[index]['Image_id'] #useful fore retrival in search phase
      poly_list.append(poly_obj)
  tree = STRtree(poly_list) #constructing R-Tree
  return grid, tree

def listit(t):
  # convert to appropriate list type 
  return list(map(listit, t)) if isinstance(t, (list, tuple)) else t



def check_polygon_in_bounds(poly, tree):
  """
  find image corrspinding to the existance of a field in the list of 
  image bounding boxes

  Args:
      poly (polygon): field
      tree (r-tree): r-tree of images

  Returns:
      List: List of intersecting images with a field
  """
  results = tree.query(poly)
  return results


def field_imageId_list(polys, count_parcels):
  """
  extract name of the intersecting polygons

  Args:
      polys (polygons): intersecting fields
      count_parcels (dict): the sanity check summary of # of fields in image ids  
  Returns:
      list: list of the image ids
  """
  list_image_ids = []
  for element in polys:
    list_image_ids.append(element.name)
    count_parcels[element.name] += 1
  return list_image_ids


def dump_shp_to_json(shape_file, grid, tree, output_json='./data/pyshp-all-2000-sentinel-new-json'):
  """
  find intersecting polygongs in the list of available images and saving the GeoJson
  
  Args:
      shape_file (polygons): fields
      grid (dict): image bounding boxes 
      tree (r-tre): r-tree of images
      output_json (str): output path of json file
  """
  #coordinate transformation
  reader = shapefile.Reader(shape_file)
  original = Proj(fiona.open(shape_file).crs)
  print(fiona.open(shape_file).crs)
  
  #list of properties of features
  fields = reader.fields[1:]
  field_names = [field[0] for field in fields]
  field_names.append('IMAGE_ID')
  
  buffer = []
  #sanity check counters
  count_parcels = defaultdict(int)
  index = 0
  num_matched = 0
  failed_projection = 0
  
  #loop through the polygo fields
  for sr in reader.iterShapeRecords():
    if index % 10000 == 0:
      print('Parsed ', index)
    index += 1
    geom = sr.shape.__geo_interface__
    shp_geom = shape(geom)
    intersect = check_polygon_in_bounds(shp_geom, tree)
    if len(intersect) != 0:
      num_matched += len(intersect)
      print("Matched:" + str(index))
      print("Number matched:", num_matched)
      
      id_list = field_imageId_list(intersect, count_parcels)
      atr = dict(zip(field_names, sr.record))
      sr.record.append(id_list)
      geom['coordinates'] = listit(geom['coordinates'])
      try: #protection at polygons that fail at projection
        if len(geom['coordinates']) == 1: #for single polygon
            counter_method1 += 1
            x,y = zip(*geom['coordinates'][0])
            lat, long = original(x, y, inverse=True) #coordinate transformation
            geom['coordinates'] = [listit(list(zip(lat, long)))]
        else: # for multipolygons
            counter_method2 += 1
            for index_coord in range(0, len(geom['coordinates'])):
              for counter in range(0,len(geom['coordinates'][index_coord])):
                x, y = geom['coordinates'][index_coord][counter]
                lat, long = original(x, y, inverse=True) #coordinate transformation
                geom['coordinates'][index_coord][counter] = [lat, long] #(long, lat)
      except:
        failed_projection =+ 1
        print(geom['coordinates'])
      buffer.append(dict(type="Feature", geometry=geom, properties=atr))
      
      
  # write the GeoJSON file
  output_json_interval = output_json + str(num_matched) + '.json'
  print("saving json \n")
  with open(output_json_interval, 'w') as geojson:
    geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2) + "\n")
    geojson.close()
    print('saved', output_json_interval)
  
  #print summary
  print('method one count:', counter_method1)
  print('method two count:', counter_method2)
  print("Number matched:", num_matched)
  print('failed count', failed_projection)


base_dir = 'data/planet/'
csv_file = base_dir + '/img_csv.csv'
shape_file = './data/RPG_2-0__SHP_LAMB93_FR-2018_2018-01-15/RPG/1_DONNEES_LIVRAISON_2018/RPG_2-0_SHP_LAMB93_FR-2018/PARCELLES_GRAPHIQUES.shp'
if os.path.exists(base_dir + 'json_polys/') == False:
    os.makedirs(base_dir + 'json_polys/')
grid, tree = read_csv(destination, original, csv_file)
dump_shp_to_json(shape_file, grid, tree, base_dir + 'json_polys/pyshp_2000_sentinel_new_json_matched_')
