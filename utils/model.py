import tensorflow as tf
from utils.layers import ENCODER, DECODER

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2, intermediate_dim=7*7*32, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.encoder = ENCODER(latent_dim=self.latent_dim, name='encoder')
        self.decoder = DECODER(intermediate_dim=self.intermediate_dim, name='decoder')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')

    @tf.function
    def sampling(self, eps=None, num_of_smaple=1):
        if eps == None:
            eps = tf.random.normal(num_of_smaple, self.latent_dim)
        return self.decoder(eps)
    
    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * tf.exp(log_var * .5)

    def call(self, inputs):
        x_mean, x_log_var = self.encoder(inputs)
        z = self.reparameterize(x_mean, x_log_var)
        outputs = self.decoder(z)
        return outputs

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = self.reparameterize(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstruction), 
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }
