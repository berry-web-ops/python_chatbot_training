import tensorflow as tf
import keras
from training import training_data_size, max_length

class Model(tf.keras.Model):
    def __init__(self):
        self.embedding_dim = 64,
        self.max_length = max_length,
        self.training_data_size = training_data_size,
        self.embed = keras.layers.Embedding(self.training_data_size, self.embedding_dim),
        self.dropout = keras.layers.Dropout(0.2),
        self.conv1 = keras.layers.Conv1D(64, 5, activation='relu'),
        self.maxpool = keras.layers.MaxPooling1D(pool_size=4),
        self.lstm = keras.layers.LSTM(64),
        self.dense1 = keras.layers.Dense(64, activation='relu'),
        self.dense2 = keras.layers.Dense(self.max_length, activation='softmax')