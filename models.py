from keras import Sequential, Input, layers
from keras.regularizers import l2
from keras.layers import Dense, BatchNormalization, ReLU, Dropout, MaxPooling2D, Conv2D, Flatten, BatchNormalization, Dropout
from config import mobilenet_v2, efficientnet_v2,inception_v3
import tensorflow_hub as hub


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, padding='valid', strides=1):
        super(CNNBlock, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size,
                           padding=padding, strides=strides)
        self.bn = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = ReLU()(x)
        return x


class ANNBlock(layers.Layer):
    def __init__(self, nodes, k_l2_reg):
        super(ANNBlock, self).__init__()
        self.dense = Dense(nodes, kernel_regularizer=l2(k_l2_reg))
        self.bn = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor)
        x = self.bn(x, training=training)
        x = ReLU()(x)
        return x


class C3Block(layers.Layer):
    def __init__(self, channels, kernel, pool_size=2, dropout=0.1):
        super(C3Block, self).__init__()
        self.cnn1 = CNNBlock(channels[0], kernel_size=kernel[0])
        self.cnn2 = CNNBlock(channels[1], kernel_size=kernel[1])
        self.cnn3 = CNNBlock(channels[1], kernel_size=kernel[2], strides=2)
        self.pool = MaxPooling2D(pool_size)
        self.drop = Dropout(dropout)

    def call(self, input_tensor, training):
        x = self.cnn1(input_tensor, training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x, training=training)
        x = self.pool(x)
        x = self.drop(x)
        return x


def cnn_model(img_height, img_width, channels):
    model = Sequential(
        [
            Input((img_height, img_width, channels)),
            C3Block([32, 32, 32], [3, 3, 5], (2, 2), 0.1),
            C3Block([64, 64, 64], [3, 3, 5], (2, 2), 0.1),
            Flatten(),
            ANNBlock(256, 0.01),
            ANNBlock(128, 0.01),
            ANNBlock(64, 0.01),
            Dropout(0.4),
            Dense(2, activation='softmax')
        ]
    )

    return model


def cnn_model2(img_height, img_width, channels):
    
    model = Sequential(
        [
            Input((img_height, img_width, channels)),
            C3Block([32, 32, 32], [3, 3, 5], (2, 2), 0.1),
            C3Block([64, 64, 64], [3, 3, 5], (2, 2), 0.1),
            C3Block([128, 128, 128], [3, 3, 5], (2, 2), 0.1),
            Flatten(),
            ANNBlock(256, 0.01),
            ANNBlock(128, 0.01),
            ANNBlock(64, 0.01),
            Dropout(0.4),
            Dense(2, activation='softmax')
        ]
    )
    
    return model

def tf_modelD1(IMAGE_SHAPE,feature_extractor_model) :
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model,
        input_shape=IMAGE_SHAPE,
        trainable=False)

    model = Sequential([
        feature_extractor_layer,
        Dense(2, activation='softmax')
    ])
    return model

def tf_modelD2(IMAGE_SHAPE,feature_extractor_model) :
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model,
        input_shape=IMAGE_SHAPE,
        trainable=False)

    model = Sequential([
        feature_extractor_layer,
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model

def tf_resnet_model(img_height, img_width, channels):
    IMAGE_SHAPE = (img_height, img_width, channels)
    feature_extractor_model = mobilenet_v2
    return tf_modelD1(IMAGE_SHAPE,feature_extractor_model)


def tf_efficientnet_model(img_height, img_width, channels):
    IMAGE_SHAPE = (img_height, img_width, channels)
    feature_extractor_model = efficientnet_v2
    return tf_modelD2(IMAGE_SHAPE,feature_extractor_model)

def tf_inception_model(img_height, img_width, channels):
    IMAGE_SHAPE = (img_height, img_width, channels)
    feature_extractor_model = inception_v3
    return tf_modelD1(IMAGE_SHAPE,feature_extractor_model)