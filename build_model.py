from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.models import Model
from keras.layers import Input, Dense


def build_model():
    inp = Input(shape=(66, 200, 3))
    x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
               activation='elu')(inp)
    x = Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2),
               activation="elu")(x)
    x = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),
               activation="elu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="elu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="elu")(x)

    x = Flatten()(x)

    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    x = Dense(1)(x)
    return Model(inputs=[inp], outputs=[x])
