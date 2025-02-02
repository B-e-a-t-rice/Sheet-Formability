import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sklearn as sk
from sklearn.model_selection import train_test_split
import pickle

print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
gpuid = 0 #int(args.gpu_id)                                                                                                                           
if gpus:
  # Restrict TensorFlow to only allocate X GB of memory on the first GPU                                                                              
  try:
    tf.config.set_visible_devices(gpus[gpuid], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpuid], True)
    '''
    tf.config.set_logical_device_configuration(
        gpus[gpuid],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
    '''
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized                                                                                   
    print(e)

inputs = np.load("all_inputs.npy")
targets = np.load("all_targets.npy")

X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)

# added more layers because original model didn't have enough downsampling
# old model had 266M params
conv_dim = 8
ff_dim = 32
k_dim = 5
pool_dim = 2
drop_rate = 0.1

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu', input_shape=(1024, 1024, 1)),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(ff_dim, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')  
])

'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu', input_shape=(1024, 1024, 1)),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
  tf.keras.layers.Dropout(drop_rate),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(ff_dim, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')  
])
'''
#  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
#  tf.keras.layers.Conv2D(conv_dim, kernel_size = (k_dim, k_dim), activation='relu'),
#  tf.keras.layers.MaxPooling2D((pool_dim, pool_dim)),
    
model.summary()


#model.compile(optimizer=Adam(learning_rate=1e-4),
#model.compile(optimizer=Adam(learning_rate=1e-2),
#model.compile(optimizer=Adam(learning_rate=1e-5),
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',  
              metrics=['accuracy'])

batchsize = 8
# buffer size is # of elements, not MB
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=batchsize*2).batch(batchsize)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_dataset = test_dataset.shuffle(buffer_size=batchsize*2).batch(batchsize)

model_checkpoint = ModelCheckpoint(
  filepath='checkpoint.model.keras',
  monitor='val_loss',
  mode='min',
  save_best_only=True)

optim = model.fit(train_dataset, 
                  epochs=100, 
                  validation_data = test_dataset,
                  callbacks=[model_checkpoint])

#model.save("trained_cnn1.h5")
model.save("final_cnn1.keras")

with open('history.pkl', 'wb') as f:
  pickle.dump(optim.history, f)

#model.evaluate(X_test, y_test)

# Predict the values from the testing dataset
#Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
#Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert testing observations to one hot vectors
#Y_true = np.argmax(y_test,axis = 1)
# compute the confusion matrix
#confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

#plt.figure(figsize=(10, 8))
#sn.heatmap(confusion_mtx, annot=True, fmt='g')
#plt.savefig("Confusion.png")




# model1 = model.fit(X_train, y_train,
                   #validation_data=(X_test, y_test),
                    #batch_size= 100,
                    #epochs= 10)


'''fig, ax = plt.subplots(2,1)
ax[0].plot(model1.history['loss'], color='b', label="Training Loss")
ax[0].plot(model1.history['val_loss'], color='r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(model1.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(model1.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.savefig("acc_loss.png")'''
