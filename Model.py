## Loading packages
import pandas as pd # Used to open CSV files 
import numpy as np # Used for matrix operations
import cv2 # Used for image augmentation
from matplotlib import pyplot as plt
np.random.seed(666)
from scipy import ndimage
from scipy import signal

# Pair down
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

# Loading data local vs kaggle
df_train = pd.read_json('/Users/kailukowiak/DATA602_Final_Project/data/processed/train.json') # this is a dataframe
#df_train = pd.read_json('../input/train.json') # this is a dataframe
# df_test = pd.read_json('../input/test.json')
df_test = pd.read_json('/Users/kailukowiak/DATA602_Final_Project/data/processed/test.json')

#df_test = pd.read_json('../input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)

# Preprocess
def pre_process_median(df):
    imgs = []
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i, row in df.iterrows():
        # Concatinate square images to one 75 x 150 pixle image. 
        concat = np.concatenate((np.reshape(np.array(df.iloc[i,0]),(75,75)), 
                   np.reshape(np.array(df.iloc[i,1]),(75,75))), axis =1)
        # Denoise image
        mean_denoised = ndimage. median_filter(concat, 5) # Five can be changed but it's a nice multiple of 75
        # Take first derivative. 
        deriv = signal.convolve2d(mean_denoised, xder, mode= 'valid')
        # Append to array
        deriv = np.absolute(deriv)
        sx = ndimage.sobel(deriv, axis=0, mode='constant')
        sy = ndimage.sobel(deriv, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        imgs.append(sob)
        #imgs.append(deriv)

    return np.array(imgs)
    
X_train = pre_process_median(df_train)
X_test = pre_process_median(df_test)

img_rows, img_cols = 73, 148
# Rescale train and test data   
######################################################### 
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



num_classes = 2
Y_train = np.array(df_train.is_iceberg)
#Y_train = keras.utils.to_categorical(Y_train, num_classes)

# def get_more_images(imgs):
    
#     more_images = []
#     vert_flip_imgs = []
#     hori_flip_imgs = []
      
#     for i in range(0,imgs.shape[0]):
#         a=imgs[i,:,:,0]
#         b=imgs[i,:,:,1]
#         c=imgs[i,:,:,2]
        
#         av=cv2.flip(a,1)
#         ah=cv2.flip(a,0)
#         bv=cv2.flip(b,1)
#         bh=cv2.flip(b,0)
#         cv=cv2.flip(c,1)
#         ch=cv2.flip(c,0)
        
#         vert_flip_imgs.append(np.dstack((av, bv, cv)))
#         hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
#     v = np.array(vert_flip_imgs)
#     h = np.array(hori_flip_imgs)
       
#     more_images = np.concatenate((imgs,v,h))
    
#     return more_images

# Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))
Xtrain = X_train
Xtest = X_test


def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(73, 148, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's view progress 
#history = model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

print(history.history.keys())


model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(X_train, Y_train, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

# df_test = pd.read_json('../input/test.json')
# df_test.inc_angle = df_test.inc_angle.replace('na',0)
## Test part of the model
#Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(X_test)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('submission.csv', index=False)
prin
