import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.model import VAE
from utils.callback import Display_sampling
from utils.utils import plot_latent_space, plot_label_clusters

ap = argparse.ArgumentParser()
ap.add_argument("--LATENT_DIM", required=False, type=int, help="latent represent dimmension")
ap.add_argument("--epoch", required=False, type=int, help="number of epoch")
ap.add_argument("--batch", required=False, type=int, help="batch size")
ap.add_argument("--lr", required=False, type=float, help="learning rate")
ap.add_argument("--fig_save", required=False, type=bool, help="Whether to save the generated image")
ap.add_argument("--model_save", required=False, type=bool, help="Whether to save the generated model")
args = ap.parse_args()

def preprocess_images(images):
    return images.reshape((images.shape[0], 28, 28, 1)) / 255.

if __name__ == '__main__':
    (train_images, y_train), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)

    LATENT_DIM = 2
    if args.LATENT_DIM:
        LATENT_DIM = args.LATENT_DIM
    INTERMEDIATE_DIM = 7*7*32

    vae = VAE(LATENT_DIM, INTERMEDIATE_DIM)
    input_arr = tf.random.uniform((1, 28, 28, 1))
    outputs = vae(input_arr)

    vae.build(train_images.shape)
    print(vae.summary())

    lr = 1e-4
    if args.lr:
        lr = args.lr

    optimizer = tf.keras.optimizers.Adam(lr)
    vae.compile(optimizer=optimizer)

    fig_save = False
    if args.fig_save:
        fig_save = args.fig_save

    test_sample = test_images[:16, :, :, :]
    sampling_callback = Display_sampling(vae, test_sample, display=False, save=fig_save, clear_output=False)

    total_epoch_count = 5
    if args.epoch:
        total_epoch_count = args.epoch
    batch_size = 512
    if args.batch_size:
        batch_size = args.batch_size

    hist = vae.fit(
        x=train_images,
        epochs=total_epoch_count,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[sampling_callback]
    )

    if args.model_save == True:
        vae.save("vae_save")