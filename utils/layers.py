import tensorflow as tf

class ENCODER(tf.keras.layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        super(ENCODER, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=3, 
            strides=(2, 2), 
            activation='relu', 
            name='encoder_conv1'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=3, 
            strides=(2, 2), 
            activation='relu',
            name='encoder_conv2'
        )
        self.flatten = tf.keras.layers.Flatten(name='encoder_flatten')
        self.dense = tf.keras.layers.Dense(2*self.latent_dim, name='encoder_dense')
        self.mean_dense = tf.keras.layers.Dense(self.latent_dim, name='encoder_mean_dense')
        self.log_var_dense = tf.keras.layers.Dense(self.latent_dim, name='encoder_log_var_dense')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.mean_dense(x)
        z_log_var = self.log_var_dense(x)
        return z_mean, z_log_var


class DECODER(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim=7*7*32, **kwargs):
        super(DECODER, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.dense = tf.keras.layers.Dense(self.intermediate_dim, name='decoder_dense')
        self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 32), name='decoder_reshape')
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv1'
        )
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='decoder_conv2'
        )
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            name='decoder_conv3'
        )

    def call(self, z):
        z = self.dense(z)
        z = self.reshape(z)
        z = self.conv1(z)
        z = self.conv2(z)
        logit = self.conv3(z)
        return logit

