"""
train_warping_net.py
-------------------------
Trains the warping network, saving the best checkpoints.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print(tf.__version__)
import sys
import os
import time
from warping_net import DM2Flux, plot_to_image

run_num = sys.argv[1]
datapath = './data/univ_000_VTFT.hdf5'

# Set up directory
baseDir = './training_runs/'
expDir = baseDir+'warpnet'+str(run_num)+'/'
if not os.path.isdir(baseDir):
    os.mkdir(baseDir)
if not os.path.isdir(expDir):
    os.mkdir(expDir)
    os.mkdir(os.path.join(expDir, 'models'))
else:
    print("Experiment directory %s already exists, exiting"%expDir)
    sys.exit()

net = DM2Flux(datapath, expDir)

bestchi = 1e15
bestL1 = 1e15
for epoch in range(net.EPOCHS):
    start = time.time()
    
    with net.train_summary_writer.as_default():
        for input_image, target in net.train_dataset:
            gen_loss = net.train_step(input_image, target)
            if tf.equal(net.generator_optimizer.iterations % net.log_freq, 0):
                # Generate sample imgs
                fig = net.generate_images(input_image, target)
                tf.summary.image("genimg", plot_to_image(fig),
                                 step=net.generator_optimizer.iterations)
                fig, chi, meanL1 = net.pix_hist()
                tf.summary.image("pixhist", plot_to_image(fig),
                                 step=net.generator_optimizer.iterations)

                # Log scalars
                tf.summary.scalar('G_loss', net.G_loss.result(), 
                                  step=net.generator_optimizer.iterations)
                tf.summary.scalar('chi', chi, step=net.generator_optimizer.iterations)
                tf.summary.scalar('meanL1', meanL1, step=net.generator_optimizer.iterations)
                net.G_loss.reset_states()
                
                # Save model if chi, L1 is good
                if net.generator_optimizer.iterations > 40000:
                    if chi < bestchi:
                        net.checkpoint.write(file_prefix = os.path.join(net.checkpoint_dir, 'BESTCHI'))
                        bestchi = chi
                        print('BESTCHI: iter=%d, chi=%f'%(net.generator_optimizer.iterations, chi))
                    if meanL1 < bestL1:
                        net.checkpoint.write(file_prefix = os.path.join(net.checkpoint_dir, 'BESTL1'))
                        bestL1 = meanL1
                        print('BESTL1: iter=%d, L1=%f'%(net.generator_optimizer.iterations, meanL1))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

print('DONE')

