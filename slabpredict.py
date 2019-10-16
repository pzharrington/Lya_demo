"""
slabpredict.py
--------------
Loads a SavedModel of the flux mapping network and runs inference on 
the full 1024x128x1024 validation set, saving the results.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import h5py

loaded = tf.saved_model.load("./trained_models/fluxmapping_SavedModel")
print('Loaded model')
print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

SIZE = 128

Ddir = './data/'
with h5py.File(Ddir+'univ_000_real.hdf5', 'r') as hf:
    sampleDM = hf['DM'][:,896:,:].astype(np.float32)
    sampleFT = hf['FT'][:,896:,:].astype(np.float32)
print('Loaded samples')

dat = np.expand_dims(sampleDM, axis=0)
dat = np.expand_dims(dat, axis=-1)
print(dat.shape)

pred = infer(tf.constant(dat))
print('DONE predicting')
pred = pred['lambda'].numpy()
print(pred.shape)
np.save(Ddir+'generated_Lya_redshift.npy', pred)
print(np.mean(np.abs(sampleFT - pred[0,:,:,:,0])))


