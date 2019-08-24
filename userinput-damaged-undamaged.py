
# coding: utf-8

# In[ ]:


from keras.models import load_model
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

model = load_model('C:\\Users\\muskan_nema\\Downloads\\gaurav\\Transfer_learning_2795_2epoch_91.8%.h5')

#validation_dir = r'C:\Users\muskan_nema\Downloads\gaurav\dataset_version_2.1_2795_images\VALIDATION'
image_dir = r'C:\Users\muskan_nema\Downloads\gaurav\test_input'

datagen = ImageDataGenerator(rescale=1./255)
#batch_size = 20
#validation_generator = datagen.flow_from_directory(
    #validation_dir,
    #target_size=(224, 224),
    #batch_size=batch_size,
    #class_mode='categorical',
    #shuffle=False)

user_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),
    #batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

fnames = user_generator.filenames
#ground_truth = validation_generator.classes
label2index = {'damaged': 0, 'undamaged': 1}
idx2label = dict((v,k) for k,v in label2index.items())
print(label2index)
predictions = model.predict_generator(user_generator, steps=user_generator.samples/user_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

print ('\n')

        


# In[55]:


import shutil 
import os 


# In[56]:


for i in range(len(predictions)):
    pred_class = np.argmax(predictions[i])
    pred_label = idx2label[pred_class]
    
    print('Prediction :{}'.format(
        pred_label,))
    
    if pred_label=='damage':
        src = r"C:\Users\muskan_nema\Downloads\gaurav\test_input\\" + fnames[i]
        dest = r"C:\Users\muskan_nema\Downloads\gaurav\damaged_images"
        shutil.copy(src,dest)


# In[4]:


print(predictions)


# In[5]:


Prediction=0
for i in range(len(predictions)):
    pred_class = np.argmax(predictions[i])
    print(pred_class)
    pred_label = idx2label[pred_class]
    Image_Title=fnames[i].split('\\')[1].split('_')[0].lower()
    Image_name=fnames[i]
    if Image_Title == pred_label:
        print('Image name:{}, Prediction :{}'.format(
        Image_name,
        pred_label))
        print ("\n")
        Prediction=Prediction+1
        original = load_img('{}/{}'.format(image_dir,fnames[i]))
        plt.imshow(original)
        plt.show()
    

print("Predictions=",Prediction)


# In[58]:


Error=0
for i in range(len(predictions)):
    pred_class = np.argmax(predictions[i])
    pred_label = idx2label[pred_class]
    #print (fnames)
    Image_Title=fnames[i].split('\\')[1].split('_')[0].lower()
    Image_name=fnames[i]
    if Image_Title != pred_label:
        print('Image name:{}, Prediction :{}'.format(
        Image_name,
        pred_label))
        print ("\n")
        Error=Error+1
        original = load_img('{}/{}'.format(image_dir,fnames[i]))
        plt.imshow(original)
        plt.show()
    
    
print("Error=",Error)


# In[59]:


Accuracy=(Prediction/user_generator.samples)*100
print("Accuracy=",Accuracy)

