import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self):
        self._setup_gpu()

    def _setup_gpu(self):
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

    def load_data(self, input_file, target_file, test_size=0.2, val_size=0.2, batch_size=32):
        ''''''
        # Load data
        with tf.device('/CPU:0'):
            inputs = np.load(input_file)
            targets = np.load(target_file)
            print(f"input shape: {inputs.shape}")

            # Train-test split
            X_trainval, X_test, y_trainval, y_test = train_test_split(inputs, targets, test_size=test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size)
            print(f"X test shape: {X_test.shape}")

        # Create datasets
        with tf.device('/CPU:0'):
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            #test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Batch and shuffle datasets
        with tf.device('/GPU:0'):
            train_dataset = train_dataset.shuffle(buffer_size=200).batch(batch_size)
            val_dataset = val_dataset.shuffle(buffer_size=200).batch(batch_size)

        return train_dataset, val_dataset, X_test, y_test