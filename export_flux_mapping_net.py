"""
export_flux_mapping_net.py
--------------------------
Loads pre-trained weights for the flux mapping network and exports a
SavedModel to be used to run inference on the validation set.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from flux_mapping_net import DM2Flux

# Build model (use dummy pathnames since we are loading pre-trained weights)
expDir = 'n/a'
datapath = 'n/a'
net = DM2Flux(datapath, expDir, resuming=True)

# Load pre-trained weights
net.checkpoint.restore(os.path.join('./trained_models', 'fluxmapping_net'))
print('Loaded succesfully')

tf.saved_model.save(net.generator, './trained_models/fluxmapping_SavedModel')
print('DONE')

