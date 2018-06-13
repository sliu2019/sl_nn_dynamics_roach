import pprint
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import IPython
import math
import matplotlib.pyplot as plt
import pickle
import threading
import multiprocessing
import os
import sys
from six.moves import cPickle
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.misc import imread


#add nn_dynamics_roach to sys.path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#my imports
from nn_dynamics_roach.msg import velroach_msg
from utils import *
from dynamics_model import Dyn_Model
from controller_class import Controller
from controller_class_playback import ControllerPlayback
###################from myalexnet_forward import returnPictureEncoding

gpu_device = 0
gpu_frac = 0.9 #0.3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
config = tf.ConfigProto(gpu_options=gpu_options,
                        log_device_placement=False,
                        allow_soft_placement=True,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)

with tf.Session(config=config) as sess:
	# CHANGE THIS FILEPATH BEFORE SCP
	restore_dynamics_model_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/run_1/models/model_epoch0.ckpt'
	# saver = tf.train.Saver()
	# saver.restore(sess, restore_dynamics_model_filepath)


	new_saver = tf.train.import_meta_graph(restore_dynamics_model_filepath + ".meta")
	#print( tf.train.latest_checkpoint('./'))
	what=new_saver.restore(sess, restore_dynamics_model_filepath)
	#print(what)
	all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#print(all_vars)
	var_vals = sess.run(all_vars)

	#print(type(var_vals))
	#print(len(var_vals))

tf.reset_default_graph()
with tf.Session(config=config) as sess2:
	
	restore_dynamics_model_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/run_1/models/model_epoch59.ckpt'
	# saver = tf.train.Saver()
	# saver.restore(sess, restore_dynamics_model_filepath)


	new_saver2 = tf.train.import_meta_graph(restore_dynamics_model_filepath + ".meta")
	#print( tf.train.latest_checkpoint('./'))
	what=new_saver2.restore(sess2, restore_dynamics_model_filepath)
	#print(what)
	all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	var_vals2 = sess2.run(all_vars)
	# saved_weights = np.load("bvlc_alexnet.npy")
	# print(type(saved_weights))

	# print(np.array_equal(var_vals, saved_weights))
	for i in range(len(var_vals)):
		print(np.array_equal(var_vals[i], var_vals2[i]))
		print(all_vars[i])
	IPython.embed()
