---
title: "Seprating Overlapping Bounding Boxes To Non-overlapping Polygon Masks"
excerpt: <br/><img src='/images/thumb3.png' width =330>
collection: portfolio
---



## Introduction

In this project, I decided to paly with the challenge of identifying objects in images using bounding boxes, which often overlap. While some models, such as Mask RCNN, can handle overlapping bounding boxes without issue, others like U-Net, a semantic segmentation model, may experience difficulties when converting these bounding boxes to masks. In these cases, separating individual objects becomes problematic.

I aim to effectively separates overlapping bounding boxes and establishes a margin between them. I'll be utilizing data from the Global Wheat Detection Dataset to accomplish this goal.


```python
import numpy as np 
import pandas as pd 
from fastai.vision import *
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, LineString
from tqdm import tqdm
```


```python
path = Path('/global-wheat-detection/')
df = pd.read_csv(path/'train.csv')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>width</th>
      <th>height</th>
      <th>bbox</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>[834.0, 222.0, 56.0, 36.0]</td>
      <td>usask_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>[226.0, 548.0, 130.0, 58.0]</td>
      <td>usask_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>[377.0, 504.0, 74.0, 160.0]</td>
      <td>usask_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>[834.0, 95.0, 109.0, 107.0]</td>
      <td>usask_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>[26.0, 144.0, 124.0, 117.0]</td>
      <td>usask_1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>147788</th>
      <td>5e0747034</td>
      <td>1024</td>
      <td>1024</td>
      <td>[64.0, 619.0, 84.0, 95.0]</td>
      <td>arvalis_2</td>
    </tr>
    <tr>
      <th>147789</th>
      <td>5e0747034</td>
      <td>1024</td>
      <td>1024</td>
      <td>[292.0, 549.0, 107.0, 82.0]</td>
      <td>arvalis_2</td>
    </tr>
    <tr>
      <th>147790</th>
      <td>5e0747034</td>
      <td>1024</td>
      <td>1024</td>
      <td>[134.0, 228.0, 141.0, 71.0]</td>
      <td>arvalis_2</td>
    </tr>
    <tr>
      <th>147791</th>
      <td>5e0747034</td>
      <td>1024</td>
      <td>1024</td>
      <td>[430.0, 13.0, 184.0, 79.0]</td>
      <td>arvalis_2</td>
    </tr>
    <tr>
      <th>147792</th>
      <td>5e0747034</td>
      <td>1024</td>
      <td>1024</td>
      <td>[875.0, 740.0, 94.0, 61.0]</td>
      <td>arvalis_2</td>
    </tr>
  </tbody>
</table>
<p>147793 rows × 5 columns</p>
</div>




```python
def bbox2mask(x):
    labels = np.array(x)
    mask = torch.zeros(1024,1024)
    for l in labels:
        mask[l[1]:l[1]+l[3], l[0]:l[0]+l[2]] = 1
    return mask

def bbox_center(x):
    labels = np.array(x)
    mask = torch.zeros(1024,1024)
    for l in labels:
        mask[(2*l[1]+l[3])//2, (2*l[0]+l[2])//2] = 1
    return mask

def box2polygon(x):
    return Polygon([(x[0], x[1]), (x[0]+x[2], x[1]), (x[0]+x[2], x[1]+x[3]), (x[0], x[1]+x[3])])
```


```python
boxes = df.groupby('image_id').agg({'bbox' : lambda x : list(x)})
box = boxes.iloc[2]
file = str(path/'train'/box.name) + '.jpg'
img = open_image(file).data.numpy().transpose(1,2,0)
bbox = np.array([eval(l) for l in box.bbox]).astype(int).tolist()
mask = bbox2mask(bbox)
gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in bbox]})
gdf.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POLYGON ((437.000 988.000, 535.000 988.000, 53...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POLYGON ((309.000 527.000, 419.000 527.000, 41...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POLYGON ((414.000 595.000, 499.000 595.000, 49...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POLYGON ((238.000 949.000, 350.000 949.000, 35...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POLYGON ((442.000 56.000, 570.000 56.000, 570....</td>
    </tr>
  </tbody>
</table>
</div>


The following function takes two bounding boxes as input, both of which are shapely Polygons, and returns the sliced region for box A


```python
def slice_box(box_A:Polygon, box_B:Polygon, margin=10, line_mult=10):
    vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
    vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
    vec_AB_norm = np.linalg.norm(vec_AB)
    split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin
    line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
    split_box = shapely.ops.split(box_A, line)
    if len(split_box) == 1: return split_box, None, line
    is_center = [s.contains(box_A.centroid) for s in split_box]
    if sum(is_center) == 0: 
        return split_box[0], None, line
    where_is_center = np.argwhere(is_center).reshape(-1)[0]
    where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]
    split_box_center = split_box[where_is_center]
    split_box_out = split_box[where_not_center]
    return split_box_center, split_box_out, line
```


```python
inter = gdf.loc[gdf.intersects(gdf.iloc[20].geometry)]

box_A = inter.iloc[0].values[0]
box_B = inter.iloc[1].values[0]
polyA, _, lineA = slice_box(box_A, box_B, margin=10, line_mult=1.2)
polyB, _, lineB = slice_box(box_B, box_A, margin=10, line_mult=1.2)

boxes = gpd.GeoDataFrame({'geometry': [box_A, box_B]})
centroids =  gpd.GeoDataFrame({'geometry': [box_A.centroid, box_B.centroid]})
splited_boxes = gpd.GeoDataFrame({'geometry': [polyA, polyB]})
lines = gpd.GeoDataFrame({'geometry': [lineA, lineB]})

fig, ax = plt.subplots(dpi=120)
boxes.plot(ax=ax, facecolor='gray', edgecolor='k', alpha=0.5)
centroids.plot(ax=ax, c='k')
ax.axis('off');

fig, ax = plt.subplots(dpi=120)
boxes.plot(ax=ax, facecolor='gray', edgecolor='k', alpha=0.1)
splited_boxes.plot(ax=ax, facecolor='olive', edgecolor='k')
centroids.plot(ax=ax, c='k')
lines.plot(ax=ax, color='k')
ax.axis('off');
```


    

    
<img src="/images/output_7_0.png">


    

<img src="/images/output_7_1.png">


```python
def intersection_list(polylist):
    r = polylist[0]
    for p in polylist:
        r = r.intersection(p)
    return r
    
def slice_one(gdf, index):
    inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]
    if len(inter) == 1: return inter.geometry.values[0]
    box_A = inter.loc[index].values[0]
    inter = inter.drop(index, axis=0)
    polys = []
    for i in range(len(inter)):
        box_B = inter.iloc[i].values[0]
        polyA, *_ = slice_box(box_A, box_B)
        polys.append(polyA)
    return intersection_list(polys)

def slice_all(gdf):
    polys = []
    for i in range(len(gdf)):
        polys.append(slice_one(gdf, i))
    return gpd.GeoDataFrame({'geometry': polys})
```


```python
res_df = slice_all(gdf)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5), dpi=120)
gdf.plot(ax=ax1, alpha=0.5, color='gray')
#gdf.plot(ax=ax2, alpha=0.1, facecolor='gray')
res_df.plot(ax=ax2, alpha=0.5, color='olive')
ax1.axis('equal')
ax2.axis('equal')
ax1.set_title('Original bounding boxes')
ax2.set_title('Splited bounding boxes')
fig.tight_layout()
```


    
    
<img src="/images/output_10_0.png">

## Rasterize polygons


```python
import rasterio.features

raster = rasterio.features.rasterize(res_df.geometry, out_shape=(1024,1024), merge_alg=rasterio.enums.MergeAlg.replace)

fig, axes = plt.subplots(ncols=2, dpi=120)
axes[0].imshow(img)
axes[0].imshow(mask, alpha=0.4)
axes[1].imshow(img)
axes[1].imshow(raster, alpha=0.4)
```




    <matplotlib.image.AxesImage at 0x7f3300c2a850>




    
<img src="/images/output_14_1.png">
    
<img src="/images/over.png" width =200>
<img src="/images/nonover.png" width =200>

## Saving new masks


```python
import PIL
import zipfile
import cv2

mask = cv2.imencode('.png', (raster*255).astype(np.uint8))[1]
boxes = df.groupby('image_id').agg({'bbox' : lambda x : list(x)})

with zipfile.ZipFile('split_masks.zip', 'w') as mask_out:
    for i in progress_bar(range(len(boxes))):
        box = boxes.iloc[i]
        file = str(path/'train'/box.name) + '.jpg'
        img = open_image(file).data.numpy().transpose(1,2,0)
        bbox = np.array([eval(l) for l in box.bbox]).astype(int).tolist()
        gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in bbox]})
        res_df = slice_all(gdf)
        raster = rasterio.features.rasterize(res_df.geometry, out_shape=(1024,1024), merge_alg=rasterio.enums.MergeAlg.replace)
        mask = cv2.imencode('.png', (raster*255).astype(np.uint8))[1]
        mask_out.writestr(f'{box.name}.png', mask)
```




