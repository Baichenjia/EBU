import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers


class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                                   padding='same', name='conv1')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                                   padding='same', name='conv2')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                   padding='same', name='conv3')
        self.fla = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(units=512, activation='relu', name='dense1')
        self.dense2 = layers.Dense(units=num_actions, activation=None, name='dense2')

    def call(self, h):
        h = tf.cast(h, tf.float32) / 255.     # (None, 84, 84, 4)
        h = self.conv1(h)                     # (None, 21, 21, 32)
        h = self.conv2(h)                     # (None, 11, 11, 64)
        h = self.conv3(h)                     # (None, 11, 11, 64)
        h = self.fla(h)                       # (None, 7744)
        h = self.dense1(h)                    # (None, 512)
        action_scores = self.dense2(h)        # (None, 4)
        return action_scores

