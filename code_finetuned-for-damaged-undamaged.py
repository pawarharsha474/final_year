
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# In[3]:


from keras.layers import merge


# In[4]:


from keras.models import load_model
import cv2
import numpy as np
from scipy.misc import imread,imresize


# In[5]:


from keras.applications.vgg16 import VGG16
vgg_conv = VGG16(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
print(vgg_conv.summary())


# In[6]:


train_dir = r'C:\Users\muskan_nema\Downloads\gaurav\dataset_version_2.1_2795_images\TRAIN'
validation_dir = r'C:\Users\muskan_nema\Downloads\gaurav\dataset_version_2.1_2795_images\VALIDATION'
#image_dir = r'D:\Gaurav\transfer learning\pumpkin\user input'

#nTrain = 1840
#nVal = 460
#nUser = 10


# In[20]:


batch_size = 20
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


# In[21]:



validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)


# In[22]:


for layer in vgg_conv.layers[:]:
    layer.trainable = False


# In[23]:


for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# In[24]:


from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()


# In[25]:


model.add(vgg_conv)


# In[26]:


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# In[27]:


model.summary()


# In[28]:


# Compile the model
import time
start=time.time()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

    
# Train the Model
history = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples/train_generator.batch_size ,
epochs=1,
validation_data=validation_generator,
validation_steps=validation_generator.samples/validation_generator.batch_size,
verbose=1)

end=time.time()
print(end-start)


# In[16]:


model.save('TEST.h5')


# In[17]:


fnames = validation_generator.filenames

ground_truth = validation_generator.classes
print(ground_truth)

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
print (label2index)


# In[18]:


predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)


# In[19]:


errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))


# In[ ]:


for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
    
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()


# In[ ]:


for i in range(len(predictions)):
    pred_class = np.argmax(predictions[i])
    pred_label = idx2label[pred_class]
    
    
    print('Original label:{}, Prediction :{}'.format(
        fnames[i],  #.split('/')[0]
        pred_label))
    print ("\n")

