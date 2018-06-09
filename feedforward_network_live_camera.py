
import numpy as np
import tensorflow as tf
from numpy import *
from pylab import *
'''
OLD NETWORK: 
	inp x 500 --> 1x500
	500 x 500 --> 1x500
	500 x out --> 1xout

NEW NETWORK:
	inp x 500 --> 1x500
	(501*5) x 500 --> 1x500
	500 x out --> 1xout

	fused:
		([u 1]' [v 1]).ravel
		(501*5)
'''

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

def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype, tiled_camera_input):
	# xdim = (227, 227, 3)
	# x = tf.placeholder(tf.float32, (None,) + xdim)

	net_data = np.load("bvlc_alexnet.npy").item()
	

	#conv1
	#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
	k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
	conv1W = tf.Variable(net_data["conv1"][0], trainable = False)
	conv1b = tf.Variable(net_data["conv1"][1], trainable = False)
	conv1_in = conv(tiled_camera_input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
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
	conv2W = tf.Variable(net_data["conv2"][0], trainable = False)
	conv2b = tf.Variable(net_data["conv2"][1], trainable = False)
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
	conv3W = tf.Variable(net_data["conv3"][0], trainable = False)
	conv3b = tf.Variable(net_data["conv3"][1], trainable = False)
	conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv3 = tf.nn.relu(conv3_in)

	#conv4
	#conv(3, 3, 384, 1, 1, group=2, name='conv4')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
	conv4W = tf.Variable(net_data["conv4"][0], trainable = False)
	conv4b = tf.Variable(net_data["conv4"][1], trainable = False)
	conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv4 = tf.nn.relu(conv4_in)


	#conv5
	#conv(3, 3, 256, 1, 1, group=2, name='conv5')
	k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
	conv5W = tf.Variable(net_data["conv5"][0], trainable = False)
	conv5b = tf.Variable(net_data["conv5"][1], trainable = False)
	conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
	conv5 = tf.nn.relu(conv5_in)

	#maxpool5
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	#fc6
	#fc(4096, name='fc6')
	fc6W = tf.Variable(net_data["fc6"][0], trainable = False)
	fc6b = tf.Variable(net_data["fc6"][1], trainable = False)
	fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

	#fc7
	#fc(4096, name='fc7')
	fc7W = tf.Variable(net_data["fc7"][0], trainable = False)
	fc7b = tf.Variable(net_data["fc7"][1], trainable = False)
	fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

	#fc8
	#fc(1000, relu=False, name='fc8')
	fc8W = tf.Variable(net_data["fc8"][0], trainable = False)
	fc8b = tf.Variable(net_data["fc8"][1], trainable = False)
	fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

	#prob
	#softmax(name='prob'))
	#prob = tf.nn.softmax(fc8)

	init = tf.initialize_all_variables()
	
	# output = sess.run(fc8, feed_dict = {x:image})

	# out = []
	# for i in output:
	# out.append(np.ndarray.flatten(i))
	# out = np.array(out)

	# np.random.seed(0)
	# matrix = np.random.uniform(low=-1, high=1,size=(len(out[0]), 5))
	# points = []
	# for i in out:
	# points.append(i.dot(matrix))

	# return points
	
	
	# **********ORIGINAL FEEDFORWARD NETWORK******************
	#vars
	intermediate_size= 200 #########depth_fc_layers
	reuse= False
	initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf_datatype)
	flatten = tf.contrib.layers.flatten
	fc = tf.contrib.layers.fully_connected

	# After AlexNet, flatten and random project
	# Do you need to flatten? It is flat already, since AlexNet contains FC for the last 3 layers
	AlexNet_out_flattened = flatten(fc8)
	random_proj_matrix = tf.random_uniform(np.array([1000, 5]), minval=-1, maxval=1, seed=0)
	AlexNet_out_projected = tf.matmul(AlexNet_out_flattened, random_proj_matrix)

	# Original feedforward_network_camera
	#1st hidden layer
	fc_1 = fc(inputState, num_outputs=intermediate_size, activation_fn=None, 
			weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
	h_1 = tf.nn.relu(fc_1)


	#pass onehot in through some small fc layers
	this_size = 32
	oh_1 = fc(AlexNet_out_projected, num_outputs=this_size, activation_fn=None, 
			weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
	oh_1 = tf.nn.relu(oh_1)
	oh_2 = fc(oh_1, num_outputs=this_size, activation_fn=None, 
			weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
	oh_2 = tf.nn.relu(oh_2)

	# Should I be using 5 here?
	oh_3 = fc(oh_2, num_outputs=5, activation_fn=None, 
			weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
	v = tf.nn.relu(oh_3)

	#fuse
		#h_1 is [bs x 500]
		#tiled_onehots is [bs x 4]
	u= tf.transpose(h_1) 
	u_batch = tf.expand_dims(u, 2) #[500 x bs x 1]
	u_batch = tf.transpose(u_batch, [1, 0, 2]) #[bs x 500 x 1]
	v_batch = tf.expand_dims(v, 1) #[bs x 1 x 4]  
	fuse = flatten(tf.matmul(u_batch, v_batch)) #[bs x 500 x 4]  --> [bs x 2000]  

	#2nd hidden layer
	fc_2 = fc(fuse, num_outputs=intermediate_size, activation_fn=None, 
			weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
	h_2 = tf.nn.relu(fc_2)

	# make output layer
	z=fc(h_2, num_outputs=outputSize, activation_fn=None, weights_initializer=initializer, 
		biases_initializer=initializer, reuse=reuse, trainable=True)
	
	return z