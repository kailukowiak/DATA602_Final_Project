# Smoothing
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage

path = os.path.expanduser('~/Data_607_Final_Project') # Your file must have the same name.
os.chdir(path) # Change this to your working directory
train = pd.read_json('~/DATA602_Final_Project/data/processed/train.json')

icebergs = train[train.is_iceberg==1]
ships = train[train.is_iceberg==0]
###################################

def reshapedDF(df):
    imgs = []
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i, row in df.iterrows():
        # Concatinate square images to one 75 x 150 pixle image. 
        concat = np.concatenate((np.reshape(np.array(train.iloc[i,0]),(75,75)), 
                   np.reshape(np.array(train.iloc[i,1]),(75,75))), axis =1)
        
        # Append to array
        imgs.append(concat)

    return np.array(imgs)
    
Xtrain = reshapedDF(train)

imgplot = plt.imshow(Xtrain[1])


import cv2


from scipy.ndimage import median_filter

smooth3 = median_filter(Xtrain[1], 5)
smooth2 = median_filter(Xtrain[1], 2)
smooth1 = median_filter(Xtrain[1], 1)

imgplot = plt.imshow(smooth1)
imgplot = plt.imshow(smooth2)
imgplot = plt.imshow(smooth3)
imgplot = plt.imshow(Xtrain[1])

# Sobels stuff:
from skimage import filter

edges = filter.sobel(Xtrain[1])
imgplot = plt.imshow(edges)

