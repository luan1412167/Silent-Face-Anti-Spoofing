import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.layers import Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization

# declare network model with channels last: NO ERROR
K.set_image_data_format('channels_first')
input = Input(shape=(1001, 1001, 3), dtype='float32')
x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=4)(x)
x = Activation('relu')(x)