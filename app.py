# Import libraries
import pandas as pd # Used to open CSV files 
import numpy as np # Used for matrix operations
import cv2 # Used for image augmentation
from matplotlib import pyplot as plt
np.random.seed(101)

# Import Keras Specific Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from scipy import signal
from scipy import ndimage
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Disables warnings in TF WRT CPU
print("Loading Data...")
#os.chdir(sys.path[0])
wd = os.getcwd()
# this is a dataframe
df_train = pd.read_json('/usr/src/app/train.json') 
#df_test = pd.read_json('test.json')
print("Filtering and resizing images...")
def get_scaled_imgs(df):
    ''' Scales, generates and puts filters on the images.
    We worked under the philosophy that images were easier to understand
    would also be easier to train.
    '''
    imgs = []
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Median filter to reduce noise
        band_1 = ndimage. median_filter(band_1, 5) # Five can be changed but it's a nice multiple of 75
        band_2 = ndimage. median_filter(band_2, 5) # Five can be changed but it's a nice multiple of 75
        band_3 = ndimage. median_filter(band_3, 5) # Five can be changed but it's a nice multiple of 75
        
        # Make a first derivative of the change
        band_1 = signal.convolve2d(band_1, xder, mode= 'valid')
        band_2 = signal.convolve2d(band_2, xder, mode= 'valid')
        band_3 = signal.convolve2d(band_3, xder, mode= 'valid')
        
        # Edge Detection
        sx_1 = ndimage.sobel(band_1, axis=0, mode='constant')
        sy_1 = ndimage.sobel(band_1, axis=1, mode='constant')
        band_1 = np.hypot(sx_1, sy_1)
        
        sx_2 = ndimage.sobel(band_2, axis=0, mode='constant')
        sy_2 = ndimage.sobel(band_2, axis=1, mode='constant')
        band_2 = np.hypot(sx_2, sy_2)
        
        sx_3 = ndimage.sobel(band_3, axis=0, mode='constant')
        sy_3 = ndimage.sobel(band_3, axis=1, mode='constant')
        band_3 = np.hypot(sx_3, sy_3)
        
        # Make RGB Channels
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

print("Generating more Data...")
# Xtrain Data
Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])

# Replacing na values on the angle.
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)


Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]


def get_more_images(imgs):
    '''
    Function to artificially generate training data
    '''
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))



def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain)) # This works becasue the two extra fake datasets are in the same order.

print('Getting the model...')

def getModel():
    '''Build a CNN for 2D images'''
    # Keras Boiler plate
    model=Sequential()
    # Laywer 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(73, 73, 3))) # 73 not 75 because of the change caused by the derevative.
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Layer 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Layer 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    #Layer 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    # Flatten Dense layers
    model.add(Flatten())
    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
print('Training model...')
model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.3)

print("Fitting Model...")
print(history.history.keys())


model.load_weights(filepath = '.mdl_wts.hdf5')
# Print out accuracy
score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

print("Making predictions...")
df_test = pd.read_json('test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print("Creating Prediction .csv...")
print(submission.head(10))

submission.to_csv('/usr/src/output/submission.csv', index=False) #/usr/src/output/ 
print("All done!")