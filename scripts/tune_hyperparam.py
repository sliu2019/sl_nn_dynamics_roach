import numpy as np 
import time
import matplotlib.pyplot as plt
import tensorflow as tf

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_action import *
from utils import *
#from run_kh_test import *
from run_kh_test2 import *
from feedforward_network_camera import feedforward_network
# Hyperparameter tuning for K (number of random offshoots) and H (length of each offshoot)
# Setting from paper is K = 500, H = 4
# We assume visualize_rviz is False. Timing may vary is visualize_rviz = True

mappings = np.load("images.npy")

gpu_device = 0
gpu_frac = 0.3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
config = tf.ConfigProto(gpu_options=gpu_options,
						log_device_placement=False,
						allow_soft_placement=True,
						inter_op_parallelism_threads=1,
						intra_op_parallelism_threads=1)

with tf.Session(config=config) as sess:	
	# Dynamic model VARS
	model_name = "camera_no_xy"
	restore_dynamics_model_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/saved_models/'+str(model_name)+'/model_aggIter0.ckpt'
	tf_datatype= tf.float32
	np_datatype= np.float32
	print_minimal=False
	use_camera = True
	use_one_hot = True

	curr_env_onehot = create_onehot('carpet', use_camera, mappings)

	N = 400

	# Why is this 6? 4 surfaces? output of camera pipeline is 5 x 1?
	one_hot_dims = 6

	# RESTORING DATA
	run_num = 1
	save_dir = save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/run_'+ str(run_num)
	dataX = np.load(save_dir+ '/data/dataX.npy')
	dataY = np.load(save_dir+ '/data/dataY.npy')
	dataZ = np.load(save_dir+ '/data/dataZ.npy')

	mean_x = np.mean(dataX, axis = 0)
	dataX = dataX - mean_x
	std_x = np.std(dataX, axis = 0)
	dataX = np.nan_to_num(dataX/std_x)

	mean_y = np.mean(dataY, axis = 0) 
	dataY = dataY - mean_y
	std_y = np.std(dataY, axis = 0)
	dataY = np.nan_to_num(dataY/std_y)

	mean_z = np.mean(dataZ, axis = 0) 
	dataZ = dataZ - mean_z
	std_z = np.std(dataZ, axis = 0)
	dataZ = np.nan_to_num(dataZ/std_z)


	# INIT DYNAMICS MODEL 
	inputSize = 23
	outputSize = 24
	lr = 0.001
	batchsize = 1000
	x_index =0
	y_index=1
	num_fc_layers = 2
	depth_fc_layers = 500


	dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, 0, x_index, y_index, 
						num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, 
						std_x, std_y, std_z, tf_datatype, np_datatype, print_minimal, feedforward_network, 
						use_one_hot, curr_env_onehot, N, one_hot_dims = one_hot_dims)
	sess.run(tf.initialize_all_variables())  
	
	# RESTORE OLD MODEL
	saver = tf.train.Saver()
	saver.restore(sess, restore_dynamics_model_filepath)

	# PARAMETERS FOR COMPUTE OPTIMAL ACTION
	full_curr_state = np.array([  9.28741917e-02,   4.23423916e-01,   6.47974089e-02,
		 3.61877245e-01,  -1.21618398e-02,   2.42200294e-02,
		 9.94444613e-01,  -1.05261157e-01,   9.95437301e-01,
		-9.54179205e-02,   9.97466056e-01,   7.11439950e-02,
		-1.22400000e+03,   6.41100000e+03,   3.87000000e+02,
		 1.04121634e-01,  -9.94564571e-01,   8.79012226e-01,
		-4.76799230e-01,   5.39300430e+01,   4.64499797e+01,
		 1.03000000e+02,  -5.40000000e+01,   8.02000000e+02])
	abbrev_curr_state = full_curr_state[3:]
	desired_states = np.array([[ -1.34932888,   0.23541372],
	   [ -0.34932888,   0.23541372],
	   [  0.65067112,   0.23541372],
	   [  1.65067112,   0.23541372],
	   [  2.65067112,   0.23541372],
	   [  3.65067112,   0.23541372],
	   [  4.65067112,   0.23541372],
	   [  5.65067112,   0.23541372],
	   [  6.65067112,   0.23541372],
	   [  7.65067112,   0.23541372],
	   [  8.65067112,   0.23541372],
	   [  9.65067112,   0.23541372],
	   [ 10.65067112,   0.23541372],
	   [ 11.65067112,   0.23541372],
	   [ 12.65067112,   0.23541372],
	   [ 13.65067112,   0.23541372],
	   [ 14.65067112,   0.23541372],
	   [ 15.65067112,   0.23541372],
	   [ 16.65067112,   0.23541372],
	   [ 17.65067112,   0.23541372],
	   [ 18.65067112,   0.23541372],
	   [ 19.65067112,   0.23541372],
	   [ 20.65067112,   0.23541372],
	   [ 21.65067112,   0.23541372],
	   [ 22.65067112,   0.23541372],
	   [ 23.65067112,   0.23541372],
	   [ 24.65067112,   0.23541372],
	   [ 25.65067112,   0.23541372],
	   [ 26.65067112,   0.23541372],
	   [ 27.65067112,   0.23541372],
	   [ 28.65067112,   0.23541372],
	   [ 29.65067112,   0.23541372],
	   [ 30.65067112,   0.23541372],
	   [ 31.65067112,   0.23541372],
	   [ 32.65067112,   0.23541372],
	   [ 33.65067112,   0.23541372],
	   [ 34.65067112,   0.23541372],
	   [ 35.65067112,   0.23541372],
	   [ 36.65067112,   0.23541372],
	   [ 37.65067112,   0.23541372]])
	left_min = 131.072
	left_max = 589.8240000000001
	right_min = 131.072
	right_max = 589.8240000000001
	optimal_action = np.array([ 472.13226039,  250.08067659])
	step = 80
	dt_steps = 1
	yaw_cos_index = 10
	yaw_sin_index = 11 

	# Setting below to None only works if visualize_rviz = False. Otherwise, they must be rospy Publishers
	publish_markers_desired = None
	publish_markers = None

	curr_line_segment = np.array(1)
	horiz_penalty_factor = 30
	backward_discouragement = 10
	heading_penalty_factor = 5
	old_curr_forward = np.array(0.40674813836812973)


	#K_list = range(250, 751, 10)
	K_list = range(250, 300, 50)
	H_list = range(1, 3)
	# SInce you're doing this on a diff computer, the approriate ranges might be completely different
	time_limit = 0.097


	print("Beginning grid search")
	max_H_for_K = []

	for K in K_list:
		print("***New K*****:", K)
		# print("dyn model params: ", inputSize, outputSize, sess, lr, batchsize, 0, x_index, y_index, 
		# 			num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, 
		# 			std_x, std_y, std_z, tf_datatype, np_datatype, print_minimal, feedforward_network, 
		# 			use_one_hot, curr_env_onehot, K, one_hot_dims)
		for H in H_list:
			print("*************NEW H***************:", H)


			a = Actions(visualize_rviz=False)

			start_time = time.time()
			a.compute_optimal_action(np.copy(full_curr_state), np.copy(abbrev_curr_state), desired_states, \
																											  left_min, left_max, right_min, right_max, \
																											  np.copy(optimal_action), step, dyn_model, K, \
																											  H, dt_steps, x_index, y_index, \
																											  yaw_cos_index, yaw_sin_index, \
																											  mean_x, mean_y, mean_z, \
																											  std_x, std_y, std_z, publish_markers_desired, \
																											  publish_markers, curr_line_segment, \
																											  horiz_penalty_factor, backward_discouragement, \
																											  heading_penalty_factor, old_curr_forward)
			time_compute_action = time.time() - start_time
			if time_compute_action > time_limit:
				max_H_for_K.append(H-1)
				break
		if time_compute_action <= time_limit:
			max_H_for_K.append(H)

	print(max_H_for_K)

	save_dir = os.path.dirname(os.path.abspath(__file__)) + '/kh_models'
	saver = tf.train.Saver(max_to_keep=0)
	save_path = saver.save(sess, save_dir+ '/camera_no_xy' + '.ckpt')

tf.reset_default_graph()

# Plotting: just for your own visualization and sanity check
# print(K_list)
# print(len(K_list))
# print(len(max_H_for_K))
save_fig_filename = "kh_plot.png"
plt.plot(K_list, max_H_for_K)
#plt.show()
plt.savefig(save_fig_filename)

# For every point on the boundary, run the robot. 
# Now using run_kh_test2! 
# KH_test_runner = KH_Test_Runner()
# empirical_avg_costs = KH_test_runner.run_roach(K_list, max_H_for_K)
# empirical_avg_costs = np.array(empirical_avg_costs)

# index = np.argmin(empirical_avg_costs)

# best_K = K_list[index]
# best_H = max_H_for_K[index]
# print("****************Results*********************\n")
# print("Best K: ", best_K, "and best H: ", best_H)

write_filename = "empirical_kh_costs"

num_runs_per_setting = 3
empirical_costs = np.empty((len(max_H_for_K), 3))
for i in range(len(max_H_for_K)):
	H = max_H_for_K[i]
	K = K_list[i]

	for j in range(num_runs_per_setting):
		print("********************Running Roach on K: ", K, " and H: ", H, " ***************************")
		empirical_cost_of_run = run_roach(K, H)
		
		while empirical_cost_of_run == None:
			#Run until we get a successful run
			print("********************Re-running Roach on K: ", K, " and H: ", H, " ***************************")
			tf.reset_default_graph()
			empirical_cost_of_run = run_roach(K, H)
		tf.reset_default_graph() # Maybe this will have to be called inside run_roach?
		empirical_costs[i, j] = empirical_cost_of_run

		# Save to file here
		np.save(write_filename, empirical_costs)

empirical_avg_costs = np.sum(empirical_costs, axis = 1)/float(num_runs_per_setting)
index = np.argmin(empirical_avg_costs)

best_K = K_list[index]
best_H = max_H_for_K[index]
print("****************Results*********************\n")
print("Best K: ", best_K, "and best H: ", best_H)

# Is this deterministic? I think not. Even on the same inputs, it depends on what other programs are running, etc. 

