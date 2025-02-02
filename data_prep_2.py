import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sklearn as sk
from sklearn.model_selection import train_test_split

print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
gpuid = 0 #int(args.gpu_id)                                                                                                                           
if gpus:
  # Restrict TensorFlow to only allocate X GB of memory on the first GPU                                                                              
  try:
    tf.config.set_visible_devices(gpus[gpuid], 'GPU')
    tf.config.set_logical_device_configuration(
        gpus[gpuid],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized                                                                                   
    print(e)

# Path to negative class (sample 2)
negative_path = 'Sheet 1/Intensity/'

# initialize storage of data and labels
images2 = []
labels2 = []

for f in os.listdir(negative_path):
    intensity_image = np.float32( Image.open(negative_path + f) )
    reshaped_intensity = np.mean(intensity_image, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    images2.append(rescaled_array)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images2.append(rescaled_tensor)
    labels2.append(0.0)
    #print('success')

negative_path = 'Sheet 3/Intensity/'

for f in os.listdir(negative_path):
    intensity_image = np.float32( Image.open(negative_path + f) )
    reshaped_intensity = np.mean(intensity_image, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    images2.append(rescaled_array)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images2.append(rescaled_tensor)
    labels2.append(0.0)
    #print('success')

# Path to positive class (sample 6) first 500 images
positive_path1 = 'Sheet_6_2/Intensity/'
images6 = []
labels6 = []

for f in os.listdir(positive_path1):
    #print(file)
    intensity_image = np.float32( Image.open(positive_path1 + f) )
    reshaped_intensity = np.mean(intensity_image, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    images6.append(rescaled_array)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images6.append(rescaled_tensor)
    labels6.append(1.0)

positive_path2 = 'Sheet 7/Intensity/'

for f in os.listdir(positive_path2):
    #print(file)
    intensity_image = np.float32( Image.open(positive_path2 + f) )
    reshaped_intensity = np.mean(intensity_image, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    images6.append(rescaled_array)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images6.append(rescaled_tensor)
    labels6.append(1.0)   


print("All data loaded")

inputs = np.concatenate([images2, images6], axis=0) #images2 + images6
targets = np.concatenate([labels2, labels6], axis=0) #labels2 + labels6

np.save("all_inputs_3.npy", inputs)
np.save("all_targets_3.npy", targets)
