
import tensorflow as tf
from tensorflow.keras import layers

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8, name="AttentionBlock", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        initializer = tf.keras.initializers.HeNormal()

        # Channel-wise attention
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.mlp = tf.keras.Sequential([
            layers.Dense(self.filters // (self.ratio*2), activation='relu', use_bias=False, kernel_initializer=initializer),
            layers.Dropout(0.2),
            layers.Dense(self.filters // self.ratio, activation='relu', use_bias=False, kernel_initializer=initializer),
            layers.Dropout(0.2),
            layers.Dense(self.filters, activation='sigmoid', use_bias=False, kernel_initializer=initializer)
        ])

        # Spatial attention block
        self.spatial = tf.keras.Sequential([
            layers.Conv2D(1, kernel_size=5, strides=1, padding='same', activation='sigmoid')

        ])



    def call(self, inputs,training=False):
        # Channel attention
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        avg_weight = self.mlp(avg_out)
        max_weight = self.mlp(max_out)
        scale = tf.nn.sigmoid(avg_weight + max_weight)
        scale = tf.reshape(scale, [-1, 1, 1, self.filters])
        x = inputs * scale


        # Spatial attention
        avg_map = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_map = tf.reduce_max(x, axis=-1, keepdims=True)
        combined = tf.concat([avg_map, max_map], axis=-1)
        attn_map = self.spatial(combined)

        return x * attn_map