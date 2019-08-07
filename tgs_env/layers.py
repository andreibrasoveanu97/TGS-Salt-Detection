import tensorflow as tf


class UnetEncodeLayer(tf.keras.layers.Layer):

    def __init__(self, kernel_size, filters, pool_size=(2, 2), dropout_value=0.25):
        super(UnetEncodeLayer, self).__init__()

        self.dropout_value = dropout_value
        self.layer1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPool2D(pool_size)
        if self.dropout_value is not None:
            self.dropout = tf.keras.layers.Dropout(self.dropout_value)

    def call(self, inputs):
        _out = self.layer1(inputs)
        _out = self.layer2(_out)
        _out_pool = self.pool(_out)
        if self.dropout_value is None:
            return _out, _out_pool
        return _out, self.dropout(_out_pool)


class UnetDecodeLayer(tf.keras.layers.Layer):

    def __init__(self, kernel_size, filters, dropout_value=0.25, strides=(2, 2)):
        super(UnetDecodeLayer, self).__init__()

        self.deconv_layer = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)
        self.dropout = tf.keras.layers.Dropout(dropout_value) if dropout_value is not None else None
        self.layer1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')

    def call(self, inputs, concat_layer):
        _out = self.deconv_layer(inputs)
        _out = tf.keras.layers.concatenate([concat_layer, _out])
        if self.dropout:
            _out = self.dropout(_out)
        _out = self.layer1(_out)
        return self.layer2(_out)


class UnetMiddleLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters):
        super(UnetMiddleLayer, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')

    def call(self, inputs):
        _out = self.layer1(inputs)
        return self.layer2(_out)
