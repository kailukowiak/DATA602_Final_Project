# File to import and process the data:

#import libraries:

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy import signal


# Import the data. 
# Your file must have the same name.
path = os.path.expanduser('~/Data_607_Final_Project') 
os.chdir(path) # Change this to your working directory
train = pd.read_json('~/DATA602_Final_Project/data/processed/train.json')

# split into icebergs and ships. (this is only useful for comparing images visually)

# icebergs = train[train.is_iceberg==1]
# ships = train[train.is_iceberg==0]


#  Functions to smoooth and reshape images.

def pre_process(df):
    imgs = []
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i, row in df.iterrows():
        # Concatinate square images to one 75 x 150 pixle image. 
        concat = np.concatenate((np.reshape(np.array(train.iloc[i,0]),(75,75)), 
                   np.reshape(np.array(train.iloc[i,1]),(75,75))), axis =1)
        # Denoise image
        gauss_denoised = ndimage.gaussian_filter(concat, 2)
        # Take first derivative. 
        deriv = signal.convolve2d(gauss_denoised, xder, mode= 'valid')
        # Append to array
        imgs.append(deriv)

    return np.array(imgs)


Xtrain = pre_process(train) # Training data
Ytrain = np.array(train['is_iceberg']) 
# Labels. (Data must remain in this order to not misclassify.)

imgplot = plt.imshow(Xtrain[5])