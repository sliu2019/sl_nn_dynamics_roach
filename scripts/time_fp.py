# This file builds the file NN (but considers AlexNet to be detached from the rest)
# WIll time how long it takes to finish a K-H loop, and tests whether you can have 2 NN using tensorflow
# To be tested on GPU (newton4): should come well under 0.1 seconds, since the data receiving takes non-zero time too


# Refactor myalexnet_forward to save the inert graph (maybe class, init function?) and then a function that evaluates it on a batch of images
import tensorflow as tf 
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
import sys, os

# My imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myalexnet_forward import *
from feedforward_network_camera import *

tf_datatype = tf.float32

g_1 = tf.Graph()
with g_1.as_default():
	# Operations created in this scope will be added to `g_1`.
	x = tf.placeholder(tf.float32, (None,) + xdim)
	fc8 = form_alexnet(x)
	init = tf.initialize_all_variables()

	gpu_device = 0
	gpu_frac = 0.9 #0.3
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
	config = tf.ConfigProto(gpu_options=gpu_options,
													log_device_placement=False,
													allow_soft_placement=True,
													inter_op_parallelism_threads=1,
													intra_op_parallelism_threads=1)
	sess_1 = tf.Session(config = config)
	sess_1.run(init)

	# Do a "priming" warm-up run
	training_mean= [123.68, 116.779, 103.939]
	s = "../images/carpet_images_1.jpg"
	im1 = (imread(s)[:,:,:3]).astype(np.float32)

	im1 = im1 - training_mean #mean(im1)
	im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

	images = []
	for i in range(1):
		images.append(im1)

	start_time = time.time()
	output = sess_1.run(fc8, feed_dict = {x:images})
	print("warm-up run took: ", time.time() - start_time)

g_2 = tf.Graph()
with g_2.as_default():
	# Operations created in this scope will be added to `g_2`.
	inputSize = 23
	outputSize = 24
	lr = 0.001
	batchsize = 1000
	x_index =0
	y_index=1
	num_fc_layers = 2
	depth_fc_layers = 500
	one_hot_dims = 6

	#placeholders
	x_ = tf.placeholder(tf_datatype, shape=[None, inputSize], name='x') #inputs
	tiled_onehots = tf.placeholder(tf_datatype, shape=[None, one_hot_dims]) #tiled one hot vectors
	curr_nn_output = feedforward_network(x_, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype, tiled_onehots)

	sess_2 = tf.Session(config = config)

	restore_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/saved_models/camera_no_xy'
	saver = tf.train.Saver()
	saver.restore(sess_2, restore_dir+ '/model_aggIter0.ckpt')
	sess_2.run(tf.initialize_all_variables())
	

	left_min = 131.072
	left_max = 589.8240000000001
	right_min = 131.072
	right_max = 589.8240000000001

	full_curr_state = np.array([  9.28741917e-02,   4.23423916e-01,   6.47974089e-02,
			 3.61877245e-01,  -1.21618398e-02,   2.42200294e-02,
			 9.94444613e-01,  -1.05261157e-01,   9.95437301e-01,
			-9.54179205e-02,   9.97466056e-01,   7.11439950e-02,
			-1.22400000e+03,   6.41100000e+03,   3.87000000e+02,
			 1.04121634e-01,  -9.94564571e-01,   8.79012226e-01,
			-4.76799230e-01,   5.39300430e+01,   4.64499797e+01,
			 1.03000000e+02,  -5.40000000e+01,   8.02000000e+02])
	abbrev_curr_state = full_curr_state[3:]
	random_actions = np.random.uniform([left_min, right_min], [left_max, right_max], (1, 2))
	inputs = np.concatenate((np.expand_dims(abbrev_curr_state, axis = 1).T, random_actions), axis=1) 


	onehots = np.zeros((1, one_hot_dims))
	onehots[0, 1] = 1 

	start_time = time.time()
	model_output = sess_2.run([curr_nn_output], feed_dict={x_: inputs, tiled_onehots: onehots}) 
	print("warmup time on dynamics model: ", time.time() - start_time)

assert fc8.graph is g_1
assert sess_1.graph is g_1

assert curr_nn_output.graph is g_2
assert sess_2.graph is g_2


# run alexnet
training_mean= [123.68, 116.779, 103.939]
s = "../images/turf_images_1.jpg"
im1 = (imread(s)[:,:,:3]).astype(np.float32)

im1 = im1 - training_mean #mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

images = []
for i in range(1):
	images.append(im1)

start_time = time.time()
output = sess_1.run(fc8, feed_dict = {x:images})
print("Total time to setup AlexNet and do FP: ", time.time() - start_time)

# Run rest of net
inputs = np.random.uniform(size=(1, 23))
onehots = np.random.uniform(size=(1, one_hot_dims))

start_time = time.time()
model_output = sess_2.run([curr_nn_output], feed_dict={x_: inputs, tiled_onehots: onehots}) 
print("time to evaluate the dynamics model: ", time.time() - start_time)

#On newton4;
"""('warm-up run took: ', 0.573228120803833)
2018-06-13 18:25:07.067512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:01:00.0, compute capability: 6.1)
('warmup time on dynamics model: ', 0.0051538944244384766)
('Total time to setup AlexNet and do FP: ', 0.003340005874633789)
('time to evaluate the dynamics model: ', 0.0008409023284912109)
"""

"""('warm-up run took: ', 0.5731780529022217)
2018-06-13 18:32:42.056410: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:01:00.0, compute capability: 6.1)
('warmup time on dynamics model: ', 0.00563502311706543)
('Total time to setup AlexNet and do FP: ', 0.003261089324951172)
('time to evaluate the dynamics model: ', 0.0009469985961914062)
"""