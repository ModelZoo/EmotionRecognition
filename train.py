import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from model import Fer2013Model
import numpy as np
import pandas as pd

tf.flags.DEFINE_string('data_dir', './fer2013/fer2013.csv', help='Data dir')
tf.flags.DEFINE_float('learning_rate', 0.001, help='Learning Rate')
tf.flags.DEFINE_integer('epochs', 1000, help='Max Epochs', allow_override=True)


class Trainer(BaseTrainer):
    
    def __init__(self):
        BaseTrainer.__init__(self)
        self.model_class = Fer2013Model
    
    def prepare_data(self):
        # read data
        path_data = self.flags.data_dir
        data = pd.read_csv(path_data)
        
        # get emotion distribution
        emotion_cat = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        target_counts = data['emotion'].value_counts().reset_index(drop=False)
        target_counts.columns = ['emotion', 'number_samples']
        target_counts['emotion'] = target_counts['emotion'].map(emotion_cat)
        print('Emotion Distribution of Data', target_counts)
        
        # split data
        data['pixels'] = data['pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
        
        data_train = data[data['Usage'] == 'Training']
        size_train = data_train.shape[0]
        print('Number samples in the training dataset: ', size_train)
        
        data_eval = data[data['Usage'] != 'Training']
        size_eval = data_eval.shape[0]
        print('Number samples in the evalelopment dataset: ', size_eval)
        
        # retrieve train input and target
        x_train, y_train = data_train['pixels'].tolist(), \
                           tf.keras.utils.to_categorical(data_train['emotion'].astype('float32'), 7)
        # reshape images to 4D (num_samples, width, height, num_channels)
        x_train = np.array(x_train, dtype='float32').reshape(-1, 48, 48, 1)
        # normalize images with max (the maximum pixel intensity is 255)
        x_train = x_train / 255.0
        
        # retrieve eval input and target
        x_eval, y_eval = data_eval['pixels'].tolist(), \
                         tf.keras.utils.to_categorical(data_eval['emotion'].astype('float32'), 7)
        # reshape images to 4D (num_samples, width, height, num_channels)
        x_eval = np.array(x_eval, dtype='float32').reshape(-1, 48, 48, 1)
        # normalize images with max
        x_eval = x_eval / 255.0
        
        print('xy train shape:', x_train.shape, y_train.shape)
        print('xy eval shape:', x_eval.shape, y_eval.shape)
        print('Sample', x_train[0], y_train[0], x_train.dtype, y_train.dtype)
        return (x_train, y_train), (x_eval, y_eval)


if __name__ == '__main__':
    Trainer().run()
