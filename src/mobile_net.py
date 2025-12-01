import tensorflow as tf
from tensorflow.keras import layers, Model

def depthwise_separable_block(x, pointwise_filters, strides=1):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def MobileNet(input_shape=(32,32,3), num_classes=10, alpha=1.0):
    inputs = layers.Input(shape=input_shape)

    # initial conv
    x = layers.Conv2D(int(32*alpha), kernel_size=3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # simplified MobileNet blocks for CIFAR-10
    filters_list = [64, 128, 128, 256, 256, 512]

    for i, f in enumerate(filters_list):
        strides = 2 if i in [1, 3] else 1  # downsampling
        x = depthwise_separable_block(x, int(f*alpha), strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model
