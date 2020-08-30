# Field Label Generation

This repo generates and process labels from polygon and sattelite imagery datasets for training convolutional neural networks for field segmentation task 

## Dataset preparation:

1. Download the French polygons dataset from 

https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/

2. Unzip the dataset under ./data directory

3. Samples random polygons. The number of polygons can be set in the script. Default is set to 2000.
```
python sample_shp.py   
```

4. Reads the shape file to gets the centroid of each polygon (Used as getting coordinates for getting satellite images such as SENTINEL-2). The centroids are the center of the satellite images.
```
python get_centroid.py   
```

5. (a) From step 5, you can extract your own satellite imagery from a public dataset (such as SENTINEL-2 or Digital Globe) and prepare a csv file containing max lat, max lon, min lat, min lon of the image. For satellite images (which comes in tfrecord format for our dataset), the following script extracts jpegs and also the csv file for the max lat, max lon, min lat, min lon of the image, useful for the next step to overlay the polygons onto image. The max lat, max lon, min lat, min lon will specify how large each satellite image spans in size. 
```
python convert_tfrecords_jpeg.py
```

5. (b) Gets only polygons that overlap in bounds of extracted images (Requires a csv-file with a unique parcel identifier, max lat, max lon, min lat, min long of each extracted image).

```
python shp2geo.py
```

6. Creates the masks (boundary and filled) of the extracted polygons and images (File input/output paths are specified in script)

```
python create_mask.py
```

7. Splits data into train/test/val

```
python split_data.py
```

8. Processes images and masks to have same dimensions if the the inputs are of different sizes

```
python train_preprocess.py
```

Reference Code: [Parcel Delineation](https://github.com/sustainlab-group/ParcelDelineation)


## Example:

![Boundary label generation](https://github.com/LobellLab/field_segmentation/blob/master/notebooks/viz_crf_overlap/11005880_bound.png)
![Area label generation](https://github.com/LobellLab/field_segmentation/blob/master/notebooks/viz_crf_overlap/11005880.png)
