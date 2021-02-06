import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from utils.model import VAE
from utils.callback import Display_sampling
from utils.utils import plot_latent_space, plot_label_clusters

ap = argparse.ArgumentParser()
ap.add_argument("--fig_save", required=False, type=bool, help="Whether to save the generated image")
ap.add_argument("--model_save", required=False, type=bool, help="Whether to save the generated model")
args = ap.parse_args()

fig_save = False
if args.fig_save:
    fig_save = args.fig_save

model_save = True
if args.model_save:
    model_save = args.model_save

def preprocess_images(images):
    return images.reshape((images.shape[0], 28, 28, 1)) / 255.

def load_mnist():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    return tf.data.Dataset.from_tensor_slices(train_images), test_images[:16, :, :, :]

def main(fig_save=False, model_save=False):
    train_images, test_sample = load_mnist()

    LATENT_DIM = config.LATENT_DIM
    MODEL_NAME = config.MODEL_NAME
    vae = VAE(LATENT_DIM, name=MODEL_NAME)
    input_arr = tf.random.uniform((1, 28, 28, 1))
    outputs = vae(input_arr)

    vae.build(train_images.shape)
    print(vae.summary())

    LEARNING_RATE = config.LEARNING_RATE
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    vae.compile(optimizer=optimizer)

    sampling_callback = Display_sampling(vae, test_sample, display=False, save=fig_save, clear_output=False)

    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    vae.fit(
        x=train_images,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[sampling_callback]
    )

    if model_save:
        vae.save("vae_save")

if __name__ == '__main__':
    main(fig_save, model_save)