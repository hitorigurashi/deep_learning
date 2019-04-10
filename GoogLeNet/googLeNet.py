from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
import keras
import sys
from inception import inception_model

def googLeNet(weight_path = None):
    input = Input(shape=(224, 224, 3))

    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                          activation='relu', kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)

    conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu',
                              kernel_regularizer=l2(0.01))(maxpool1_3x3_s2)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                          activation='relu', kernel_regularizer=l2(0.01))(conv2_3x3_reduce)
    maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)

    inception_3a = inception_model(maxpool2_3x3_s2, 64, 96, 128, 16, 32, 32)
    inception_3b = inception_model(inception_3a, 128, 128, 192, 32, 96, 64)

    maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_3b)

    inception_4a = inception_model(maxpool3_3x3_s2, 192, 96, 208, 16, 48, 64)
    inception_4b = inception_model(inception_4a, 160, 112, 224, 24, 64, 64)
    inception_4c = inception_model(inception_4b, 128, 128, 256, 24, 64, 64)
    inception_4d = inception_model(inception_4c, 112, 144, 288, 32, 64, 64)
    inception_4e = inception_model(inception_4d, 256, 160, 320, 32, 128, 128)

    maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_4e)

    inception_5a = inception_model(maxpool4_3x3_s2, 256, 160, 320, 32, 128, 128)
    inception_5b = inception_model(inception_5a, 384, 192, 384, 48, 128, 128)

    averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7, 7), padding='same')(inception_5b)
    drop1 = Dropout(rate=0.4)(averagepool1_7x7_s1)

    linear = Dense(units=1000, activation='softmax', kernel_regularizer=l2(0.01))(keras.layers.core.Flatten()(drop1))
    last = linear
    model = Model(input=input, outputs=last)
    model.summary()

if __name__ == "__main__":
    googLeNet()
