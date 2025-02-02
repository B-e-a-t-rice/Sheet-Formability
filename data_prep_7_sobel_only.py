
from PIL import Image 
import numpy as np
import matplotlib.image as mpimg
import os
import cv2



def extract_data(path, intensity_list, label):
    data = []
    labels = []
    for i in range(len(intensity_list)):
        intensity_image = cv2.imread(path + "Intensity/" + intensity_list[i])
        down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)

        # get sobel values
        sobelx_values = cv2.Sobel(src=down_samp, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        reshaped_sobel = np.mean(sobelx_values, axis=-1)
        sobelx_array = np.expand_dims(reshaped_sobel, axis=-1)
        max_sobel = np.max(reshaped_sobel)
        min_sobel = np.min(reshaped_sobel)
        rescaled_sobel = (sobelx_array-min_sobel)/(max_sobel-min_sobel)

        data.append(rescaled_sobel)
        labels.append(label)
    return data,labels


# initialize storage of data and labels for negative class
# Path to negative class
negative_path1 = 'Sheet 1/'
intensity_list = os.listdir(negative_path1+"Intensity")
data1,labels1 = extract_data(negative_path1,intensity_list,0.0)
print('success')

negative_path3 = 'Sheet 3/'
intensity_list = os.listdir(negative_path3+"Intensity")
data3,labels3 = extract_data(negative_path3,intensity_list,0.0)
print("success")

negative_data = np.concatenate([data1,data3])
negative_labels = np.concatenate([labels1,labels3])
print(negative_data.shape)
print(negative_labels.shape)


# Path to positive class 
positive_path1 = 'Sheet_6_2/'
intensity_list = os.listdir(positive_path1+"Intensity")
data6,labels6 = extract_data(positive_path1,intensity_list,1.0)
print("success")

positive_path2 = 'Sheet 7/'
intensity_list = os.listdir(positive_path2+"Intensity")
data7,labels7 = extract_data(positive_path2,intensity_list,1.0)
print("success")

positive_data = np.concatenate([data6,data7])
positive_labels = np.concatenate([labels6,labels7])
print(positive_data.shape)
print(positive_labels.shape)
print("All data loaded")

inputs = np.concatenate([negative_data, positive_data], axis=0) 
targets = np.concatenate([negative_labels, positive_labels], axis=0) 

np.save("all_inputs_with_sobel_only512.npy", inputs)
np.save("all_targets_with_sobel_only512.npy", targets)

