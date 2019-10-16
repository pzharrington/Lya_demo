"""
flux_mapping_net.py
-------------------
Defines an object that contains the necessary networks, hyperparameters, and
helper routines for the 3D flux mapping network.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import h5py
import io


def rotate_xy(x, rand):
    """Rotations and transposes across x and y axes (keep z unchanged)"""
    return tf.case(
        {tf.equal(rand, 1): lambda: x[:, ::-1, ::-1, :, :],
         tf.equal(rand, 2): lambda: x[:, :, ::-1, :, :],
         tf.equal(rand, 3): lambda: x[:, ::-1, :, :, :],
         tf.equal(rand, 4): lambda: x,
         tf.equal(rand, 5): lambda: tf.transpose(x, (0, 2, 1, 3, 4)),
         tf.equal(rand, 6): lambda: tf.transpose(x, (0, 2, 1, 3, 4))[:, ::-1, ::-1, :, :],
         tf.equal(rand, 7): lambda: tf.transpose(x, (0, 2, 1, 3, 4))[:, ::-1, :, :, :],
         tf.equal(rand, 8): lambda: tf.transpose(x, (0, 2, 1, 3, 4))[:, :, ::-1, :, :]}, default = lambda: x, exclusive = True)


def randrot_xy(input_image, real_image):
    """Wrapper for the random rotations and transposes"""
    rand = tf.random.uniform(shape=(),minval=1,maxval=8, dtype=tf.dtypes.int32)
    return rotate_xy(input_image, rand), rotate_xy(real_image, rand)


def load_image_train(input_image, real_image):
    input_image = tf.expand_dims(tf.cast(input_image, tf.float32), axis=-1)
    real_image = tf.expand_dims(tf.cast(real_image, tf.float32), axis=-1)
    return input_image, real_image


def load_image_test(input_image, real_image):
    input_image = tf.expand_dims(tf.cast(input_image, tf.float32), axis=-1)
    real_image = tf.expand_dims(tf.cast(real_image, tf.float32), axis=-1)
    return input_image, real_image


def downsample(filters, size, apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.ReLU())
    return result


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Useful for saving plots into tensorboard."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


# Data generator for tf.dataset pipeline (takes the large HDF5 file as argument)
class DataGenerator:
    """Data generator for tf.dataset pipeline (takes the large HDF5 file as argument) """
    def __init__(self, file, isTrain=True):
        self.file = file
        self.isTrain = isTrain
        self.Nsamples = 1000

    def __call__(self):
        size = 128
        with h5py.File(self.file, 'r') as hf:
            # Random crops, setting one slab aside for validation
            if self.isTrain:
                for i in range(self.Nsamples):
                    x = np.random.randint(low=0, high=1024-size)
                    y = np.random.randint(low=0, high=896-size)
                    z = np.random.randint(low=0, high=1024-size)
                    yield (hf['DM'][x:x+size, y:y+size, z:z+size], hf['FT'][x:x+size, y:y+size, z:z+size])
            else:
                for i in range(self.Nsamples):
                    x = np.random.randint(low=0, high=1024-size)
                    y = 896 
                    z = np.random.randint(low=0, high=1024-size)
                    yield (hf['DM'][x:x+size, y:, z:z+size], hf['FT'][x:x+size, y:, z:z+size])


class DM2Flux:
    
    def __init__(self, datapath, expDir, resuming=False):

        # Initialize hyperparmeters
        self.LAMBDA = 3000
        self.log_freq = 500
        self.learn_rate = 5E-5
        self.EPOCHS = 200
        self.expDir = expDir
        self.OUTPUT_CHANNELS = 1
        self.hdf5_path = datapath

        # Build networks
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learn_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learn_rate, beta_1=0.5)

        # Setup checkpointing and data pipeline
        self.checkpoint_dir = os.path.join(expDir,'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        self.stage_data_pipeline()

        # Setup tensorboard stuff
        if not resuming:
            self.train_summary_writer = tf.summary.create_file_writer(os.path.join(expDir, 'logs'))
            self.prep_summaries()

        
    def stage_data_pipeline(self):
        self.train_dataset = tf.data.Dataset.from_generator(DataGenerator(self.hdf5_path, isTrain=True), 
                                                            (tf.float32, tf.float32),
                                                            (tf.TensorShape([128,128,128]),
                                                             tf.TensorShape([128,128,128])))
        self.train_dataset = self.train_dataset.map(load_image_train)
        self.train_dataset = self.train_dataset.batch(1)
        self.train_dataset = self.train_dataset.map(randrot_xy)
        self.test_dataset = tf.data.Dataset.from_generator(DataGenerator(self.hdf5_path, isTrain=False), 
                                                           (tf.float32, tf.float32),
                                                           (tf.TensorShape([128,128,128]),
                                                            tf.TensorShape([128,128,128])))
        self.test_dataset = self.test_dataset.map(load_image_test)
        self.test_dataset = self.test_dataset.batch(1)


    def Generator(self):
        down_stack = [
            downsample(64, 4),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4)
          ]
        up_stack = [
            upsample(512, 4),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4)
          ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv3DTranspose(1, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='relu')
        concat = tf.keras.layers.Concatenate()
        inputs = tf.keras.layers.Input(shape=[None,None,None,1])
        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])
        x = last(x)
        x = tf.keras.layers.Lambda(lambda y: tf.tanh(y))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, None, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])

        down1 = downsample(64, 4)(x)
        down2 = downsample(128, 4, apply_batchnorm=True)(down1)
        down3 = downsample(256, 4, apply_batchnorm=True)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)
        conv = tf.keras.layers.Conv3D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)

        last = tf.keras.layers.Conv3D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + self.LAMBDA*l1_loss
        return total_gen_loss, gan_loss, self.LAMBDA*l1_loss
    

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            gen_loss, gan_loss, l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.G_loss.update_state(gen_loss)
        self.G_loss_gan.update_state(gan_loss)
        self.G_loss_L1.update_state(l1_loss)
        self.D_loss.update_state(disc_loss)

        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))


    def prep_summaries(self):
        self.G_loss = tf.keras.metrics.Mean(name='G_loss', dtype=tf.float32)
        self.G_loss_gan = tf.keras.metrics.Mean(name='G_loss_gan', dtype=tf.float32)
        self.G_loss_L1 = tf.keras.metrics.Mean(name='G_loss_L1', dtype=tf.float32)
        self.D_loss = tf.keras.metrics.Mean(name='D_loss', dtype=tf.float32)  

            
    def generate_images(self):
        """Visualize middle slice of the generated cubes"""
        for inp, target in self.test_dataset.take(1):
            test_input, tar = inp, target
        prediction = self.generator(test_input)
        fig = plt.figure(figsize=(15,15))
        display_list = [test_input[0], 
                        tar[0], 
                        (tar[0] - prediction[0]), 
                        prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Residual', 'Predicted Image']

        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.title(title[i])
            if i==0:
                img = display_list[i].numpy()
                img = np.squeeze(img[32,:,:])*0.5 +0.5
                plt.imshow(img, cmap='Blues')
            elif i==2:
                img = display_list[i].numpy()
                cbar = plt.imshow(np.squeeze(img[32,:,:]), cmap='seismic',
                                  norm=Normalize(vmin=-1., vmax=1.))
                plt.colorbar(fraction=0.046, pad=0.04)
            else:
                img = display_list[i].numpy()
                plt.imshow(np.squeeze(img[32,:,:]), cmap='viridis')
            plt.axis('off')
        return fig
        
    
    def flux_PDF(self):
        """Plot the flux PDF and compare to test set, return the chi-square score and the MAE"""
        gens = []
        tars = []
        for inp, tar in self.test_dataset.take(50):
            img = self.generator(inp, training=True)
            gens.append(img)
            tars.append(tar)
        gens = np.concatenate(gens, axis=0)
        tars = np.concatenate(tars, axis=0)

        tar_hist, bin_edges = np.histogram(tars, bins=50)
        gen_hist, _ = np.histogram(gens, bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig = plt.figure()
        plt.errorbar(centers, tar_hist, yerr=np.sqrt(tar_hist), fmt='ks--', label='real')
        plt.errorbar(centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
        plt.xlabel('F = exp(-tau_red)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()
        return (fig,
               np.sum(np.divide(np.power(tar_hist - gen_hist, 2.0), tar_hist)),
               np.mean(np.abs(tars - gens)))


