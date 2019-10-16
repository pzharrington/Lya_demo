"""
warping_net.py
-------------------
Defines an object that contains the necessary networks, hyperparameters, and
helper routines for the 2D warping network.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import h5py
import sys
import io


def load_image_train(input_image, real_image):
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    if np.random.uniform(low=0., high=1.0) < 0.5:
        input_image = tf.image.flip_up_down(input_image)
        real_image = tf.image.flip_up_down(real_image)
    return input_image, real_image


def load_image_test(input_image, real_image):
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def downsample(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
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


class DataGenerator:
    """Data generator used to randomly crop samples from big HDF5 file and feed 
       into a tf.dataset pipeline"""
    def __init__(self, file, isTrain=True):
        self.file = file
        self.isTrain = isTrain
        self.Nsamples = 14000

    def __call__(self):
        size = 128
        with h5py.File(self.file, 'r') as hf:
            if self.isTrain:
                for i in range(self.Nsamples):
                    x = np.random.randint(low=0, high=1024)
                    y = np.random.randint(low=0, high=896-size)
                    yield (hf['VT'][x, y:y+size,:], hf['FT'][x, y:y+size,:])
            else:
                for i in range(self.Nsamples):
                    x = np.random.randint(low=0, high=1024-size)
                    y = 896
                    yield (hf['VT'][x,y:,:], hf['FT'][x,y:,:])


class DM2Flux:
    
    def __init__(self, datapath, expDir, resuming=False):

        # Load hyperparmeters
        self.log_freq = 500
        self.learn_rate = 2E-4
        self.EPOCHS = 24
        self.expDir = expDir
        self.OUTPUT_CHANNELS = 1
        self.hdf5_path = datapath
        
        # Build networks
        self.generator = self.Generator()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learn_rate, beta_1=0.5)

        # Setup checkpointing and data pipeline
        self.checkpoint_dir = os.path.join(expDir,'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         generator=self.generator)
        self.stage_data_pipeline()

        # Setup tensorboard stuff
        if not resuming:
            self.train_summary_writer = tf.summary.create_file_writer(os.path.join(expDir, 'logs'))
            self.prep_summaries()

        
    def stage_data_pipeline(self):
        self.train_dataset = tf.data.Dataset.from_generator(DataGenerator(self.hdf5_path, isTrain=True), 
                                                            (tf.float32, tf.float32),
                                                            (tf.TensorShape([128,1024,2]),
                                                             tf.TensorShape([128,1024,1])))
        self.train_dataset = self.train_dataset.map(load_image_train)
        self.train_dataset = self.train_dataset.batch(1)
        self.test_dataset = tf.data.Dataset.from_generator(DataGenerator(self.hdf5_path, isTrain=False), 
                                                           (tf.float32, tf.float32),
                                                           (tf.TensorShape([128,1024,2]),
                                                            tf.TensorShape([128,1024,1])))
        self.test_dataset = self.test_dataset.map(load_image_test)
        self.test_dataset = self.test_dataset.batch(1)


    def Generator(self):
        down_stack = [
            downsample(64, 4),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4)
          ]

        up_stack = [
            upsample(256, 4),
            upsample(256, 4),
            upsample(256, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4)
          ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='relu')
        concat = tf.keras.layers.Concatenate()
        inputs = tf.keras.layers.Input(shape=[None,None,2])
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
        x = tf.tanh(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    
    def generator_loss(self, gen_output, target):
        return tf.reduce_mean(tf.abs(target - gen_output))

    
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input_image, training=True)
            gen_loss = self.generator_loss(gen_output, target)
            self.G_loss.update_state(gen_loss)
        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        return gen_loss


    def prep_summaries(self):
        self.G_loss = tf.keras.metrics.Mean(name='G_loss', dtype=tf.float32)
            
            
    def generate_images(self, test_input, tar):
        prediction = self.generator(test_input, training=True)
        fig = plt.figure(figsize=(15,15))
        display_list = [test_input[0,:,:,0], 
                        tar[0], 
                        prediction[0],
                        (tar[0] - prediction[0])]
        title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Residual']
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.title(title[i])
            if i==0:
                img = display_list[i].numpy()
                img = np.squeeze(img)*0.5 +0.5
                plt.imshow(img, cmap='viridis')
            elif i==3:
                img = display_list[i].numpy()
                plt.imshow(np.squeeze(img), cmap='seismic',
                                  norm=Normalize(vmin=-1., vmax=1.))
            else:
                img = display_list[i].numpy()
                plt.imshow(np.squeeze(img), cmap='viridis')
            plt.axis('off')
        return fig
        
    
    
    def pix_hist(self):
        """Plot the flux PDF and compare to test set, return the chi-square score and the MAE"""
        gens = []
        tars = []
        for inp, tar in self.test_dataset.take(100):
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


