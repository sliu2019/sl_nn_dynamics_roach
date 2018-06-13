################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import time

#from sklearn.manifold import TSNE

import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:


#net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
		'''From https://github.com/ethereon/caffe-tensorflow
		'''
		c_i = input.get_shape()[-1]
		assert c_i%group==0
		assert c_o%group==0
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
		
		
		if group==1:
				conv = convolve(input, kernel)
		else:
				input_groups = tf.split(input, group, 3)
				kernel_groups = tf.split(kernel, group, 3)
				output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
				conv = tf.concat(output_groups, 3)
		return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def form_alexnet():
	net_data = load("bvlc_alexnet.npy").item()
	x = tf.placeholder(tf.float32, (None,) + xdim)


	#conv1
	#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
	k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
	conv1W = tf.Variable(net_data["conv1"][0])
	conv1b = tf.Variable(net_data["conv1"][1])
	conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
	conv1 = tf.nn.relu(conv1_in)

	#lrn1
	#lrn(2, 2e-05, 0.75, name='norm1')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn1 = tf.nn.local_response_normalization(conv1,
																										depth_radius=radius,
																										alpha=alpha,
																										beta=beta,
																										bias=bias)

	#maxpool1
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


	#conv2
	#conv(5, 5, 256, 1, 1, group=2, name='conv2')
	k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
	conv2W = tf.Variable(net_data["conv2"][0])
	conv2b = tf.Variable(net_data["conv2"][1])
	conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv2 = tf.nn.relu(conv2_in)


	#lrn2
	#lrn(2, 2e-05, 0.75, name='norm2')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn2 = tf.nn.local_response_normalization(conv2,
																										depth_radius=radius,
																										alpha=alpha,
																										beta=beta,
																										bias=bias)

	#maxpool2
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	#conv3
	#conv(3, 3, 384, 1, 1, name='conv3')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
	conv3W = tf.Variable(net_data["conv3"][0])
	conv3b = tf.Variable(net_data["conv3"][1])
	conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv3 = tf.nn.relu(conv3_in)

	#conv4
	#conv(3, 3, 384, 1, 1, group=2, name='conv4')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
	conv4W = tf.Variable(net_data["conv4"][0])
	conv4b = tf.Variable(net_data["conv4"][1])
	conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv4 = tf.nn.relu(conv4_in)


	#conv5
	#conv(3, 3, 256, 1, 1, group=2, name='conv5')
	k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
	conv5W = tf.Variable(net_data["conv5"][0])
	conv5b = tf.Variable(net_data["conv5"][1])
	conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv5 = tf.nn.relu(conv5_in)

	#maxpool5
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	#fc6
	#fc(4096, name='fc6')
	fc6W = tf.Variable(net_data["fc6"][0])
	fc6b = tf.Variable(net_data["fc6"][1])
	fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

	#fc7
	#fc(4096, name='fc7')
	fc7W = tf.Variable(net_data["fc7"][0])
	fc7b = tf.Variable(net_data["fc7"][1])
	fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

	#fc8
	#fc(1000, relu=False, name='fc8')
	fc8W = tf.Variable(net_data["fc8"][0])
	fc8b = tf.Variable(net_data["fc8"][1])
	fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

	#prob
	#softmax(name='prob'))
	#prob = tf.nn.softmax(fc8)
	init = tf.initialize_all_variables()

	return init, fc8

def run(images):
	init, fc8 = form_alexnet()
	gpu_device = 0
	gpu_frac = 0.9 #0.3
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
	config = tf.ConfigProto(gpu_options=gpu_options,
													log_device_placement=False,
													allow_soft_placement=True,
													inter_op_parallelism_threads=1,
													intra_op_parallelism_threads=1)
	sess = tf.Session(config = config)
	sess.run(init)
	output = sess.run(fc8, feed_dict = {x:images})
	### TESTING MANY LAYERS
	# start_time = time.time()
	# output = sess.run(fc6, feed_dict = {x:image})
	# print("Time to do forward pass on fc6: ", time.time()-start_time)


	# A = np.random.rand(227, 227, 3)
	# for i in range(20):
	# 	np.tile(A, (400, 1))

	# for i in range(10):
	# 	start_time = time.time()
	# 	output = sess.run(fc6, feed_dict = {x:image})
	# 	print("Time to do forward pass on fc6: ", time.time()-start_time)
	# # start_time = time.time()
	# output = sess.run(fc7, feed_dict = {x:image})
	# print("Time to do forward pass on fc7: ", time.time()-start_time)

	# start_time = time.time()
	# output = sess.run(fc8, feed_dict = {x:image})
	# print("Time to do forward pass on fc8: ", time.time()-start_time)

	out = []
	for i in output:
		out.append(np.ndarray.flatten(i))
	out = np.array(out)

	np.random.seed(0)
	matrix = np.random.uniform(low=-1, high=1,size=(len(out[0]), 5))
	points = []
	for i in out:
		points.append(i.dot(matrix))

	return points

def timeFP():
	# tests time of forward pass
	# Result; for N = 400, it takes about 24 seconds
	training_mean= [123.68, 116.779, 103.939]
	s = "../images/carpet_images_1.jpg"
	im1 = (imread(s)[:,:,:3]).astype(float32)

	im1 = im1 - training_mean #mean(im1)
	im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
	
	images = []
	for i in range(100):
		images.append(im1)
	start_time = time.time()
	run(images)
	print("Total time to setup AlexNet and do FP: ", time.time() - start_time)

def savePictures():
	training_mean= [123.68, 116.779, 103.939]

	surfaces = ["carpet", "gravel", "turf", "styrofoam"]
	images = []
	for surf in surfaces:
		for i in range(10):
			s = "../images/" + surf + "_images_" + str(i) + ".jpg"
			im1 = (imread(s)[:,:,:3]).astype(float32)

			im1 = im1 - training_mean #mean(im1)
			im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
			images.append(im1)

	#np.save("images.npy", run(images))
	all_features = run(images)

	all_features=np.array(all_features)

	carpet_features = all_features[:10,:]
	sty_features = all_features[10:20,:]
	gravel_features = all_features[20:30,:]
	turf_features = all_features[30:40,:]

	mean_carpet_features = np.mean(carpet_features,axis=0)
	mean_sty_features = np.mean(sty_features,axis=0)
	mean_gravel_features = np.mean(gravel_features,axis=0)
	mean_turf_features = np.mean(turf_features,axis=0)

	std_carpet_features = np.std(carpet_features,axis=0)
	std_sty_features = np.std(sty_features,axis=0)
	std_gravel_features = np.std(gravel_features,axis=0)
	std_turf_features = np.std(turf_features,axis=0)


	######normalize
	all_features = all_features-np.mean(all_features)
	all_features = all_features/np.std(all_features)
	np.save("images.npy", all_features)

#savePictures()
#timeFP()



