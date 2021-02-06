from IPython import display

import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import imageio


class Display_sampling(tf.keras.callbacks.Callback):
    def __init__(self, model, test_sample, display=True, save=True, clear_output=True):
        self.model = model
        self.test_sample = test_sample
        self.display = display
        self.save = save
        self.clear_output = clear_output

    def generate_and_images(self, model, num, batch=False):
        mean, log_var = model.encoder(self.test_sample)
        z = model.reparameterize(mean, log_var)
        reconstruction = model.decoder(z)
        fig = plt.figure(figsize=(4, 4))
        for i in range(0, reconstruction.shape[0], 2):
            plt.subplot(4, 4, i+1)
            plt.imshow(reconstruction[i, :, :, 0], cmap='gray')
            plt.axis('off')

            plt.subplot(4, 4, i + 2)
            plt.imshow(self.test_sample[i, :, :, 0], cmap='gray')
            plt.axis('off')


        if batch and self.save:
            fig.suptitle(f'image_at_batch_{num}')
            plt.savefig(f'./img/batch/image_at_batch_{num}.png')
            plt.close()
        elif self.save:
            fig.suptitle(f'image_at_epoch_{num}')
            plt.savefig(f'./img/image_at_epoch_{num}.png')

        if self.display:
            plt.show()
                
    def on_train_begin(self, logs=None):
        if self.clear_output:
            display.clear_output(wait=False)
        self.generate_and_images(self.model, num=0)

    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            self.generate_and_images(self.model, batch+1, batch=True)

    def on_epoch_end(self, epoch, logs):
        if self.clear_output:
            display.clear_output(wait=False)
        self.generate_and_images(self.model, epoch+1)
            
    def on_train_end(self, logs):
        if self.save and self.display:
            anim_file = './img/vae.gif'

            with imageio.get_writer(anim_file, mode='I') as writer:
                filenames = glob.glob('./img/batch/image_at_batch_*.png')
                filenames = sorted(filenames)
                image = imageio.imread('./img/image_at_epoch_0.png')
                writer.append_data(image)
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)