import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import os
import cv2



def extract_data(path, intensity_list, label):
    data = []
    labels = []
    for i in range(len(intensity_list)):
        intensity_image = cv2.imread(path + "test_images/" + intensity_list[i])
        down_samp = cv2.resize(intensity_image, (512,512), interpolation=cv2.INTER_LINEAR)
        reshaped_intensity = np.mean(down_samp, axis=-1) 
        image_array = np.expand_dims(reshaped_intensity, axis=-1)
        rescaled_array = image_array/255.0
        #print(rescaled_array.shape)   

        # get sobel values
        #sobelx_values = cv2.Sobel(src=down_samp, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        #reshaped_sobel = np.mean(sobelx_values, axis=-1)
        #sobelx_array = np.expand_dims(reshaped_sobel, axis=-1)
        #max_sobel = np.max(reshaped_sobel)
        #min_sobel = np.min(reshaped_sobel)
        #rescaled_sobel = (sobelx_array-min_sobel)/(max_sobel-min_sobel)

        # combine sobel and intensity
        #combined_data = np.concatenate([rescaled_array,rescaled_sobel],axis=-1)
        data.append(rescaled_array)
        labels.append(label)
    return data,labels



negative_path1 = 'Sheet 5/'
intensity_list = os.listdir(negative_path1+"test_images")
data5,labels5 = extract_data(negative_path1,intensity_list,0.0)
print('success')


# Path to positive class 
positive_path1 = 'Sheet 4/'
intensity_list = os.listdir(positive_path1+"test_images")
data4,labels4 = extract_data(positive_path1,intensity_list,1.0)
print("success")


print("All data loaded")

#"I" stands for intensity "IH" stands for intensity and height "IS" stands for intensity and sobel
np.save("negative_inputs_I512_testing.npy", data5)
np.save("negative_targets_I512_testing.npy", labels5)

np.save("positive_inputs_I512_testing.npy", data4)
np.save("positive_targets_I512_testing.npy", labels4)




