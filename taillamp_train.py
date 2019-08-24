
# coding: utf-8

# In[1]:


'''
author Gaurav - This program creates and trains the model for recognizing in image whether
the Bumper is damaged or not, and will tell if Bumper is not present in image.

'''

#importing required libraries

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


#importing VGG16 pre-trained model

from keras.applications.vgg16 import VGG16

#Loading config file to read all path to resources
from configparser import ConfigParser
import yaml
config = yaml.load(open('../../../../resources/config/config.yml'))



vgg_conv = VGG16(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
print(vgg_conv.summary())


# In[6]:


#Adding Training Directory for the training of model
train_dir = config['Model']['train_dir_taillamp']



#Adding Validation Directory for the training of model
validation_dir = config['Model']['validation_dir_taillamp']


# In[7]:


#Generating and reshaping the images from the Directory

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


# In[8]:



validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)


# In[9]:


#Freezing all the previous layers of the vgg16 model
for layer in vgg_conv.layers[:]:
    layer.trainable = False


# In[10]:


#Importing models from the keras library
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()


# In[11]:



model.add(vgg_conv)


# In[12]:


#adding additional layers to the vgg16 as per our model requirement
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))


# In[13]:


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=3,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)


# In[14]:


#Save the model for the future use
model.save(config['Model']['trained_model_taillamp'])


# In[ ]:


#The below part is optional and for the validation purpose
#verify the validation accuracy of the model
#Getting image class name and Filename from the validation folder
fnames = validation_generator.filenames

ground_truth = validation_generator.classes

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
print (label2index)


# In[ ]:


#prediction the model using validation images
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)


# In[ ]:


#Collecting error images from the predictions of the model
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))


# In[ ]:


#Displaying error images with incorrect predictions
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
        fnames[i].split('/')[0],
        pred_label))
    print ("\n")

