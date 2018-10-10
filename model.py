from model_zoo.model import BaseModel
import tensorflow as tf
import numpy as np


class Fer2013Model(BaseModel):
    def __init__(self, config):
        super(Fer2013Model, self).__init__(config)
        self.num_features = 64
        # layer1
        self.conv11 = tf.keras.layers.Conv2D(filters=self.num_features, kernel_size=(3, 3), activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv12 = tf.keras.layers.Conv2D(filters=self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop1 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer2
        self.conv21 = tf.keras.layers.Conv2D(filters=2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn21 = tf.keras.layers.BatchNormalization()
        self.conv22 = tf.keras.layers.Conv2D(filters=2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn22 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer3
        self.conv31 = tf.keras.layers.Conv2D(filters=2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn31 = tf.keras.layers.BatchNormalization()
        self.conv32 = tf.keras.layers.Conv2D(filters=2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn32 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop3 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer4
        self.conv41 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.bn41 = tf.keras.layers.BatchNormalization()
        self.conv42 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.bn42 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = tf.keras.layers.Dropout(rate=0.5)
        
        # flatten
        self.flatten = tf.keras.layers.Flatten()
        
        # dense
        self.dense1 = tf.keras.layers.Dense(2 * 2 * 2 * self.num_features, activation='relu')
        self.drop5 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(2 * 2 * self.num_features, activation='relu')
        self.drop6 = tf.keras.layers.Dropout(0.4)
        self.dense3 = tf.keras.layers.Dense(2 * self.num_features, activation='relu')
        self.drop7 = tf.keras.layers.Dropout(0.5)
        
        self.dense4 = tf.keras.layers.Dense(7, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
        # layer1
        x = self.conv11(inputs)
        x = self.conv12(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        # layer2
        x = self.conv21(x)
        x = self.bn21(x, training=training)
        x = self.conv22(x)
        x = self.bn22(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        # layer3
        x = self.conv31(x)
        x = self.bn31(x, training=training)
        x = self.conv32(x)
        x = self.bn32(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        # layer4
        x = self.conv41(x)
        x = self.bn41(x, training=training)
        x = self.conv42(x)
        x = self.bn42(x, training=training)
        x = self.pool4(x)
        x = self.drop4(x, training=training)
        # flatten
        x = self.flatten(x)
        # dense
        x = self.dense1(x)
        x = self.drop5(x, training=training)
        x = self.dense2(x)
        x = self.drop6(x, training=training)
        x = self.dense3(x)
        x = self.drop7(x, training=training)
        x = self.dense4(x)
        return x
    
    def optimizer(self):
        return tf.train.AdamOptimizer(self.config.get('learning_rate'))
    
    def init(self):
        self.compile(optimizer=self.optimizer(),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    
    def infer(self, test_data, batch_size=None):
        logits = self.predict(test_data)
        preds = np.argmax(logits, axis=-1)
        return logits, preds
