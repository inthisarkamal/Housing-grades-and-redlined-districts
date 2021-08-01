#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import shapefile as shp
import pandas as pd
from pandas.api.types import CategoricalDtype

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import zipfile
import os

from ds100_utils import run_linear_regression_test

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

from IPython.display import Image


# Plot settings
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12


# In[22]:


#first we loaded the Cook County training data. We used the training data over the sale data
#because the training data includes Sale Prices. We wanted to map the most expensive homes and
#least expensive homes in the county on a map of Cook County with the historical redlining
#districts to see if there was any correlation.
with zipfile.ZipFile('cook_county_data.zip') as item:
    item.extractall()


# In[23]:


cook_county = pd.read_csv("cook_county_train.csv", index_col='Unnamed: 0')


# In[24]:


#Looking at the 300 most expensive homes in the county. We picked the top 300 homes because
#there seems to be a huge jump and range in sale prices among the top 300. When we tried to 
#use a sample size greater than 300, our map started to make incoherent patterns.
cook_county_top300=cook_county.sort_values('Sale Price', ascending=False).head(300)
cook_county_top300['Sale Price']


# In[25]:


geometry=[Point(xy) for xy in zip(cook_county_top300.Longitude, cook_county_top300.Latitude)]


# In[26]:


cook_county_top300['geometry'] = geometry


# In[27]:


#Incorporated the shape file from Mapping Inequality (https://dsl.richmond.edu/panorama/redlining/#loc=11/41.641/-87.733)
#This loaded a base map.
#Used this resource to learn how to map onto shape files: https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972
cook_map = gpd.read_file('cartodb-query.shp')
fig,ax = plt.subplots(figsize = (15,15))
cook_map.plot(ax = ax)
geometry=[Point(xy) for xy in zip(cook_county_top300.Longitude, cook_county_top300.Latitude)]
geo_df = gpd.GeoDataFrame(geometry = geometry)
print(geo_df)
g = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'Top 300')
plt.show()


# In[28]:


#Built an actual map since the different districts couldn't be seen on the base map above.
#Code was taken from this source: https://stackoverflow.com/questions/63644131/how-to-use-geopandas-to-plot-latitude-and-longitude-on-a-more-detailed-map-with
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point


ward = gpd.read_file('cartodb-query.shp', bbox=None, mask=None, rows=None)
geo_df = gpd.GeoDataFrame(geometry = geometry)

ward.crs = {'init':"epsg:4326"}
geo_df.crs = {'init':"epsg:4326"}

# plot the polygon
ax = ward.plot(alpha=0.35, color='#d66058', zorder=1)
# plot the boundary only (without fill), just uncomment
#ax = gpd.GeoSeries(ward.to_crs(epsg=3857)['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)
ax = gpd.GeoSeries(ward['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)

# plot the marker
ax = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'Top 300', zorder=3)

ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.show()


# In[29]:


#Imported an image of the redlined districts to see the different historically redlined areas.
from IPython import display
display.Image("Redlining.png")

#Source: https://digitalchicagohistory.org/files/original/1868c31f074d1ace5e7cac8f5851d60d.png


# In[30]:


#Manually plotted the location of where the markers are most concentrated in the base map onto
#the redlining map. We see that the most expensive homes are concentrated in one area that was
#historically desirable and another area which was historically redlined
from PIL import Image
from pylab import *


im = array(Image.open('Redlining.png'))

# plot the image
imshow(im)

# some points
x = [1320,1700,1680,1670,1660, 1650]
y = [700,700,650,550,500, 480]

plot(x,y,'ro')

# add title and show the plot
title('Present day most-expensive homes mapped on districts from the 1930s')
show()


# In[31]:


#Similar to above, looked at the lowest valued homes.
cook_county_bottom300=cook_county.sort_values('Sale Price', ascending=False).tail(300)
cook_county['Sale_Price']=cook_county['Sale Price']
cook_county['Sale_Price']


# In[32]:


#There seem to be a lot of null values of 1. I am curious to see which locations these null values fall in
cook_county_null= cook_county['Sale Price']==1
cook_county_null_1=cook_county[cook_county_null]
cook_county_null_1


# In[33]:


#Mapped all the null values and their corresponding locations on the map. 
#Can see that this is a huge are. However notice that the northern part of the City of Chicago
#does not have any null values. Can speak to better data quality in that area. Also happens 
#to be where the wealthier houses are.
cook_map = gpd.read_file('cartodb-query.shp')
fig,ax = plt.subplots(figsize = (15,15))
cook_map.plot(ax = ax)
geometry3=[Point(xy) for xy in zip(cook_county_null_1.Longitude, cook_county_null_1.Latitude)]
geo_df = gpd.GeoDataFrame(geometry = geometry3)
print(geo_df)
g = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*')
plt.show()


# In[35]:


ward = gpd.read_file('cartodb-query.shp', bbox=None, mask=None, rows=None)
geo_df = gpd.GeoDataFrame(geometry = geometry3)

ward.crs = {'init':"epsg:4326"}
geo_df.crs = {'init':"epsg:4326"}

# plot the polygon
ax = ward.plot(alpha=0.35, color='#d66058', zorder=1)

ax = gpd.GeoSeries(ward['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)

# plot the marker
ax = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = '', zorder=3)

ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.show()


# In[36]:


#Had to clean the data a little, thousands of data points had a null entry represented by "1".
#Removed Null entries.
cook_county_clean=cook_county[cook_county.Sale_Price !=1]
cook_county_clean['Sale_Price']


# In[37]:


#Sorted according to sale price from least expensive.
cook_county_clean_bottom300=cook_county_clean.sort_values('Sale_Price', ascending=False).tail(300)
cook_county_clean_bottom300['Sale_Price']


# In[38]:


#Similar to the first section, loaded the base map and plotted the least expensive homes onto it.
#Data shows that the spread is greater than the top 300 (huge property valuation disparity)
cook_map = gpd.read_file('cartodb-query.shp')
fig,ax = plt.subplots(figsize = (15,15))
cook_map.plot(ax = ax)
geometry2=[Point(xy) for xy in zip(cook_county_clean_bottom300.Longitude, cook_county_clean_bottom300.Latitude)]
geo_df = gpd.GeoDataFrame(geometry = geometry2)
print(geo_df)
g = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'Top 300')
plt.show()


# In[39]:


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point


ward = gpd.read_file('cartodb-query.shp', bbox=None, mask=None, rows=None)
geo_df = gpd.GeoDataFrame(geometry = geometry2)

ward.crs = {'init':"epsg:4326"}
geo_df.crs = {'init':"epsg:4326"}

# plot the polygon
ax = ward.plot(alpha=0.35, color='#d66058', zorder=1)

ax = gpd.GeoSeries(ward['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)

# plot the marker
ax = geo_df.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'Bottom 300', zorder=3)

ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.show()


# In[40]:


from IPython import display
display.Image("Redlining.png")


# In[41]:


#See that least expensive homes are in red and yellow areas but not anywhere green or blue which
#were the historically desirable areas.
from PIL import Image
from pylab import *

# read image to array
im = array(Image.open('Redlining.png'))

# plot the image
imshow(im)

# some points
x = [1250,1220,1220,1220,1350,1300,1250,1200,900,1100,1000,1050,1000]
y = [750,700,850,800,1200,1100,1250,1300,500,600,1000,1200,750]

# plot the points with red star-markers
plot(x,y,'ro')

# add title and show the plot
title('Present day least expensive homes mapped on districts from the 1930s')
show()


# In[ ]:




