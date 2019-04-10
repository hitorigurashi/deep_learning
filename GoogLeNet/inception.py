from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.regularizers import l2

def inception_model(input, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool):
    conv_1x1 = Conv2D(filters=filter_1x1, kernel_size=(1, 1), padding='same', activation='relu',kernel_regularizer=l2(0.01))(input)

    conv_3x3_reduce = Conv2D(filters=filter_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=l2(0.01))(input)
    conv_3x3 = Conv2D(filters=filter_3x3, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2(0.01))(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters=filter_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=l2(0.01))(input)
    conv_5x5 = Conv2D(filters=filter_3x3, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2(0.01))(conv_5x5_reduce)

    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filter_pool, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          kernel_regularizer=l2(0.01))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)

    return inception_output

