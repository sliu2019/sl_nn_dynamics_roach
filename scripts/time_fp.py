# This file builds the file NN (but considers AlexNet to be detached from the rest)
# WIll time how long it takes to finish a K-H loop, and tests whether you can have 2 NN using tensorflow
# To be tested on GPU (newton4): should come well under 0.1 seconds, since the data receiving takes non-zero time too


# Refactor myalexnet_forward to save the inert graph (maybe class, init function?) and then a function that evaluates it on a batch of images
import tensorflow as tf 
import numpy as np
import time

# My imports
from myalexnet_forward import *
from feedforward_network_camera import *

AlexNet = tf.Graph()
with AlexNet.as_default():
  # Operations created in this scope will be added to `g_1`.
  init, fc8 = form_alexnet()

  #c = tf.constant("Node in g_1")
  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

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
	
	#placeholders
    x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') #inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #labels
        self.next_z_ = tf.placeholder(tf_datatype, shape=[None, 3, self.outputSize], name='next_z')
        #forward pass
        if self.use_one_hot and self.use_camera:
            self.tiled_camera_input = tf.placeholder(tf_datatype, shape=[None, 227, 227, 3])
            self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize, 
                                                    num_fc_layers, depth_fc_layers, tf_datatype, self.tiled_camera_input)

	dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, 0, x_index, y_index, 
						num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, 
						std_x, std_y, std_z, tf_datatype, np_datatype, print_minimal, feedforward_network, 
						use_one_hot, curr_env_onehot, N, one_hot_dims = one_hot_dims)
	sess.run(tf.initialize_all_variables())  
  output = feefeedforward_network(x_, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype, tiled_onehots)
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
