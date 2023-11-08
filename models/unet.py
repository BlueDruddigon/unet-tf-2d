import tensorflow as tf
from keras.layers import BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dropout, Input, MaxPool2D
from keras.models import Model


def double_conv(x, n_filters, dropout_rate: float = 0.1):
    x = Conv2D(n_filters, 3, activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, 3, activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x


def upconv_pad_concat(x, y, n_filters):
    x = Conv2DTranspose(n_filters, 2, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    pad_x = y.shape[2] - x.shape[2]
    pad_y = y.shape[1] - x.shape[1]
    x = tf.pad(x, paddings=[[0, 0], [pad_y // 2, pad_y - pad_y//2], [pad_x // 2, pad_x - pad_x//2], [0, 0]])
    x = concatenate([y, x])
    return x


def get_model(img_size, n_classes, n_channels: int = 1, n_filters: int = 64):
    inputs = Input(shape=(img_size, img_size, n_channels))
    
    # Downsample - 64
    conv1 = double_conv(inputs, n_filters)  # residual
    down1 = MaxPool2D(2)(conv1)
    
    # Downsample - 128
    n_filters *= 2
    conv2 = double_conv(down1, n_filters)  # residual
    down2 = MaxPool2D(2)(conv2)
    
    # Downsample - 256
    n_filters *= 2
    conv3 = double_conv(down2, n_filters)  # residual
    down3 = MaxPool2D(2)(conv3)
    
    # Downsample - 512
    n_filters *= 2
    conv4 = double_conv(down3, n_filters)  # residual
    down4 = MaxPool2D(2)(conv4)
    
    # bottleneck - 1024
    n_filters *= 2
    conv5 = double_conv(down4, n_filters)
    
    # Upsample - 512
    n_filters /= 2
    merge6 = upconv_pad_concat(conv5, conv4, n_filters)
    
    # Upsample - 256
    n_filters /= 2
    merge7 = upconv_pad_concat(merge6, conv3, n_filters)
    
    # Upsample - 128
    n_filters /= 2
    merge8 = upconv_pad_concat(merge7, conv2, n_filters)
    
    # Upsample - 64
    n_filters /= 2
    merge9 = upconv_pad_concat(merge8, conv1, n_filters)
    
    # output projection
    outputs = Conv2D(n_classes, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(merge9)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
