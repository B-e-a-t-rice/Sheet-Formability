import tensorflow as tf
from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import os
import cv2

print(tf.__version__)



# Path to negative class
negative_path = 'Sheet 1/'
intensity_list = os.listdir(negative_path+"Intensity")
height_list = os.listdir(negative_path+"Height")

# initialize storage of data and labels
data0 = []
labels0 = []
for i in range(len(intensity_list)):
#for f in os.listdir(negative_path+"Intensity"):
    # get the intensity
    intensity_image = np.float32( Image.open(negative_path + "Intensity/" + intensity_list[i]))
    down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)
    reshaped_intensity = np.mean(down_samp, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    #print(rescaled_array.shape)   

    # get heights
    raw_heights = pd.read_csv(negative_path+"Height/"+height_list[i],skiprows=18)
    raw_heights = raw_heights.drop(['DataLine','Unnamed: 1025'],axis=1)
    np_heights = np.array(raw_heights)
    down_heights = cv2.resize(np_heights, (512,512), interpolation=cv2.INTER_LINEAR)
    max_height = np.max(down_heights)
    exp_height = np.expand_dims(down_heights,axis=-1)
    rescaled_height = exp_height/max_height
    #print(rescaled_height.shape)

    # combine height and intensity
    #combined_data = np.concatenate([rescaled_array,rescaled_height],axis=-1)


    data0.append(rescaled_height)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images2.append(rescaled_tensor)
    labels0.append(0.0)
print('success')

negative_path = 'Sheet 3/'
intensity_list = os.listdir(negative_path+"Intensity")
height_list = os.listdir(negative_path+"Height")



for i in range(len(intensity_list)):
#for f in os.listdir(negative_path+"Intensity"):
    # get the intensity
    intensity_image = np.float32( Image.open(negative_path + "Intensity/" + intensity_list[i]))
    down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)
    reshaped_intensity = np.mean(down_samp, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    #print(rescaled_array.shape)   

    # get heights
    raw_heights = pd.read_csv(negative_path+"Height/"+height_list[i],skiprows=18)
    raw_heights = raw_heights.drop(['DataLine','Unnamed: 1025'],axis=1)
    np_heights = np.array(raw_heights)
    down_heights = cv2.resize(np_heights, (512,512), interpolation=cv2.INTER_LINEAR)
    max_height = np.max(down_heights)
    exp_height = np.expand_dims(down_heights,axis=-1)
    rescaled_height = exp_height/max_height
    #print(rescaled_height.shape)

    # combine height and intensity
    #combined_data = np.concatenate([rescaled_array,rescaled_height],axis=-1)


    data0.append(rescaled_height)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images2.append(rescaled_tensor)
    labels0.append(0.0)
print('success')

# Path to positive class (sample 6) first 500 images
positive_path1 = 'Sheet_6_2/'
intensity_list = os.listdir(positive_path1+"Intensity")
height_list = os.listdir(positive_path1+"Height")

data1 = []
labels1 = []

for i in range(len(intensity_list)):
#for f in os.listdir(negative_path+"Intensity"):
    # get the intensity
    intensity_image = np.float32( Image.open(positive_path1 + "Intensity/" + intensity_list[i]))
    down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)
    reshaped_intensity = np.mean(down_samp, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    #print(rescaled_array.shape)   

    # get heights
    raw_heights = pd.read_csv(positive_path1+"Height/"+height_list[i],skiprows=18)
    raw_heights = raw_heights.drop(['DataLine','Unnamed: 1025'],axis=1)
    np_heights = np.array(raw_heights)
    down_heights = cv2.resize(np_heights, (512,512), interpolation=cv2.INTER_LINEAR)
    max_height = np.max(down_heights)
    exp_height = np.expand_dims(down_heights,axis=-1)
    rescaled_height = exp_height/max_height
    #print(rescaled_height.shape)

    # combine height and intensity
    #combined_data = np.concatenate([rescaled_array,rescaled_height],axis=-1)


    data1.append(rescaled_height)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images6.append(rescaled_tensor)
    labels1.append(1.0)
print("success")

positive_path2 = 'Sheet 7/'

intensity_list = os.listdir(positive_path2+"Intensity")
height_list = os.listdir(positive_path2+"Height")


for i in range(len(intensity_list)):
#for f in os.listdir(negative_path+"Intensity"):
    # get the intensity
    intensity_image = np.float32( Image.open(positive_path2+ "Intensity/" + intensity_list[i]))
    down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)
    reshaped_intensity = np.mean(down_samp, axis=-1) 
    image_array = np.expand_dims(reshaped_intensity, axis=-1)
    rescaled_array = image_array/255.0
    #print(rescaled_array.shape)   

    # get heights
    raw_heights = pd.read_csv(positive_path2+"Height/"+height_list[i],skiprows=18)
    raw_heights = raw_heights.drop(['DataLine','Unnamed: 1025'],axis=1)
    np_heights = np.array(raw_heights)
    down_heights = cv2.resize(np_heights, (512,512), interpolation=cv2.INTER_LINEAR)
    max_height = np.max(down_heights)
    exp_height = np.expand_dims(down_heights,axis=-1)
    rescaled_height = exp_height/max_height
    #print(rescaled_height.shape)

    # combine height and intensity
    combined_data = np.concatenate([rescaled_array,rescaled_height],axis=-1)


    data1.append(rescaled_height)
    #print(file)
    #image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    #rescaled_tensor = image_tensor/255.0
    #images6.append(rescaled_tensor)
    labels1.append(1.0)
print("success")


print("All data loaded")

inputs = np.concatenate([data0, data1], axis=0) 
targets = np.concatenate([labels0, labels1], axis=0) 

np.save("all_inputs_height.npy", inputs)
np.save("all_targets_height.npy", targets)
