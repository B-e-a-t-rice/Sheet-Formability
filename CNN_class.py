import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

class CNN_3:
    def __init__(self, input_shape, conv_dim=16, ff_dim=32, k_dim=3, pool_dim=5, pooling_type='max',drop_rate=0.0):
        '''Takes input_shape, conv_dim,ff_dim,k_dim,pool_dim,pooling_type and drop_rate
        It is necessary to pass in the input shape when initializing the function'''
        self.input_shape = input_shape
        self.conv_dim = conv_dim
        self.ff_dim = ff_dim
        self.k_dim = k_dim
        self.pool_dim = pool_dim
        self.pooling_type = pooling_type
        self.drop_rate = drop_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(self.conv_dim, kernel_size=(self.k_dim, self.k_dim), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(self.pool_dim, self.pool_dim) if self.pooling_type == 'max' else tf.keras.layers.AveragePooling2D(self.pool_dim, self.pool_dim),
            tf.keras.layers.Dropout(self.drop_rate),
            tf.keras.layers.Conv2D(self.conv_dim, kernel_size=(self.k_dim, self.k_dim), padding='valid'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(self.pool_dim, self.pool_dim) if self.pooling_type == 'max' else tf.keras.layers.AveragePooling2D(self.pool_dim, self.pool_dim),
            tf.keras.layers.Dropout(self.drop_rate),
            tf.keras.layers.Conv2D(self.conv_dim, kernel_size=(self.k_dim, self.k_dim), padding='valid'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(self.pool_dim, self.pool_dim) if self.pooling_type == 'max' else tf.keras.layers.AveragePooling2D(self.pool_dim, self.pool_dim),
            tf.keras.layers.Dropout(self.drop_rate),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dropout(self.drop_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def compile_model(self, learning_rate=0.00001, loss='binary_crossentropy', metrics=['accuracy']):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, train_data, validation_data, epochs=1000, batch_size=32,patience=20,early_stopping=True,checkpoint_path=None):
        callbacks = []

        # Add early stopping callback if enabled
        if early_stopping == True:
            early_stop = EarlyStopping(
                monitor="val_loss", 
                patience=patience,  
                mode="min", 
                restore_best_weights=True
                )
            callbacks.append(early_stop)

        # Add model checkpoint callback if a path is provided
        if checkpoint_path:
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
            callbacks.append(checkpoint_callback)

        return self.model.fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size,callbacks=callbacks)

    def evaluate_model(self, X_test,y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)
    
    def my_save_weights(self, path):
        return self.model.save_weights(path)