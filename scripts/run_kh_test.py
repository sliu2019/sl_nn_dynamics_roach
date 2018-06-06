# This file has basically the same code as train_dynamics.py, except in a more convenient form for continuous testing

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


#datatypes
tf_datatype= tf.float32
np_datatype= np.float32

mappings = np.load("images.npy")

class KH_Test_Runner(object):
	"""Based on train_dynamics, but allows you to easily do multiple runs with different K, H values."""
	def __init__(self):
		self.yaw_cos_index = 10
		self.yaw_sin_index = 11

	def run_roach(self, K_list, max_H_for_K):
		##################################
		######### SPECIFY VARS ###########
		##################################

		# Which trajectory, saving filenames
		run_num= 1                                         #directory for saving everything
		desired_shape_for_traj = "straight"                     #straight, left, right, circle_left, zigzag, figure8
		save_run_num = 0
		traj_save_path= desired_shape_for_traj + str(save_run_num)     #directory name inside run_num directory

		#######TRAINING########## 
		train_now = False

		# train_now = False: which saved model to potentially load from
		model_name = 'camera_no_xy'     #onehot_smaller, combined, camera
		
		# train_now = True: select training data
		use_existing_data = True #Basically, if true, use pre-processed data; false, re-pre-process the data specified below
		# use_existing_data = true. Specify task between: 'carpet','styrofoam', 'gravel', 'turf', 'all'
		task_type=['all']                 
		months = ['01','02']
		data_path = os.path.abspath(os.path.join(os.getcwd(), "../data_collection/"))

		# Doesn't use_one_hot have to be the negation of use_camera? If so, why are there 2 variables?
		# Use one hot is if you're going to have a conditioned NN or not; use camera is if it's going to be 1-hot or camera
		use_one_hot= True #True
		use_camera = True #True
		#Cheating method: what you do when no camera live-feed
		curr_env_onehot = create_onehot('carpet', use_camera, mappings)

		# training/validation split
		training_ratio = 0.9

		nEpoch_initial = 50
		nEpoch = 20
		state_representation = "exclude_x_y" #["exclude_x_y", "all"]
		num_fc_layers = 2
		depth_fc_layers = 500
		batchsize = 1000
		lr = 0.001

		###########TESTING############
		#which setting to run in


		#PID (velocity) vs PWM (thrust)
		use_pid_mode = True      
		slow_pid_mode = True

		#xbee connection port
		serial_port = '/dev/ttyUSB1'

		#controller
		visualize_rviz=True   #turning this off could make things go faster
		if(use_one_hot):
			N=400
		else:
			N=500
		horizon = 5 #4
		frequency_value=10
		playback_mode = False

		#length of controller run
		#num_steps_per_controller_run=50
		if(desired_shape_for_traj=='straight'):
			self.num_steps_per_controller_run=10
			if (task_type==['gravel']):
				self.num_steps_per_controller_run=85
		elif(desired_shape_for_traj=='left'):
			self.num_steps_per_controller_run= 160
			if(task_type==['turf']):
				self.num_steps_per_controller_run=110
		elif(desired_shape_for_traj=='right'):
			self.num_steps_per_controller_run= 150
			if ('gravel' in task_type):
				self.num_steps_per_controller_run=130
		elif(desired_shape_for_traj=='zigzag'):
			self.num_steps_per_controller_run=160
			if(task_type==['turf']):
				self.num_steps_per_controller_run=210
		else:
			self.num_steps_per_controller_run=0


		##############################################
		##### DONT NEED TO MESS WITH THIS PART #######
		##############################################

		#aggregation
		fraction_use_new = 0.5
		num_aggregation_iters = 1
		num_trajectories_for_aggregation= 1
		rollouts_forTraining = num_trajectories_for_aggregation

		baud_rate = 57600
		DEFAULT_ADDRS = ['\x00\x01']

		one_hot_dims=4
		if(use_camera):
			one_hot_dims=6

		#read in the training data
		path_lst = []
		for subdir, dirs, files in os.walk(data_path):
			lst = subdir.split("/")[-1].split("_")
			if len(lst) >= 3:
				surface = lst[0]
				month = lst[2]
				if ((surface in task_type or "all" in task_type) and month in months):
					for file in files:
						path_lst.append(os.path.join(subdir, file))
		path_lst.sort()
		print "num of rollouts: ", len(path_lst)/2
		training_rollouts = int(len(path_lst)*training_ratio)
		if training_rollouts%2 != 0:
			training_rollouts -= 1
		validation_rollouts = len(path_lst) - training_rollouts

		##################################
		######### MOTOR LIMITS ###########
		##################################

		#set min and max
		left_min = 1200
		right_min = 1200
		left_max = 2000
		right_max = 2000

		if(use_pid_mode):
		  if(slow_pid_mode):
			left_min = 2*math.pow(2,16)*0.001
			right_min = 2*math.pow(2,16)*0.001
			left_max = 9*math.pow(2,16)*0.001
			right_max = 9*math.pow(2,16)*0.001
		  else: #this hasnt been tested yet
			left_min = 4*math.pow(2,16)*0.001
			right_min = 4*math.pow(2,16)*0.001
			left_max = 12*math.pow(2,16)*0.001
			right_max = 12*math.pow(2,16)*0.001
		
		##################################
		######### LOG DIRECTORY ##########
		##################################

		#directory from which to get training data
		# data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + filename_trainingdata

		#directories for saving data
		save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/run_'+ str(run_num)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
			os.makedirs(save_dir+'/losses')
			os.makedirs(save_dir+'/models')
			os.makedirs(save_dir+'/data')
			os.makedirs(save_dir+'/saved_forwardsim')
			os.makedirs(save_dir+'/saved_trajfollow')
			os.makedirs(save_dir+'/'+traj_save_path)
		if not os.path.exists(save_dir+'/'+traj_save_path):
			os.makedirs(save_dir+'/'+traj_save_path)


		#return

		###############restore_dynamics_model_filepath = save_dir+ '/models/model_aggIter0.ckpt'
		self.restore_dynamics_model_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/saved_models/'+str(model_name)+'/model_aggIter0.ckpt'
		if(train_now==False):
			print("restoring dynamics model from: ", self.restore_dynamics_model_filepath) 

		##############################
		### init vars 
		##############################

		visualize_True = True
		visualize_False = False
		noise_True = True
		noise_False = False
		dt_steps= 1
		self.x_index=0
		self.y_index=1
		z_index=2

		make_aggregated_dataset_noisy = True
		make_training_dataset_noisy = True
		perform_forwardsim_for_vis= True
		print_minimal=False

		noiseToSignal = 0
		if(make_training_dataset_noisy):
			noiseToSignal = 0.01

		# num_rollouts_val = len(validation_rollouts)
		num_rollouts_val = validation_rollouts


		#################################################
		### save a file of param values to the run directory
		#################################################

		'''param_file = open(save_dir + '/params.txt', "w")
		param_file.write("\ntrain_separate_nns = " + str(train_separate_nns))
		param_file.close()'''

		#################################################
		### set GPU options for TF
		#################################################

		self.gpu_device = 0
		self.gpu_frac = 0.3
		os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_frac)
		self.config = tf.ConfigProto(gpu_options=self.gpu_options,
								log_device_placement=False,
								allow_soft_placement=True,
								inter_op_parallelism_threads=1,
								intra_op_parallelism_threads=1)

		with tf.Session(config=self.config) as sess:

			#################################################
			### read in training dataset
			#################################################

			if(use_existing_data):
				#training data
				dataX = np.load(save_dir+ '/data/dataX.npy')
				dataY = np.load(save_dir+ '/data/dataY.npy')
				dataZ = np.load(save_dir+ '/data/dataZ.npy')

				print("Dimensions of dataX are: ", dataX.shape)

				if(use_one_hot):
					dataOneHots = np.load(save_dir+ '/data/dataOneHots.npy')
				else:
					dataOneHots=0

				#validation data
				states_val = np.load(save_dir+ '/data/states_val.npy')
				controls_val = np.load(save_dir+ '/data/controls_val.npy')

				if(use_one_hot):
					onehots_val = np.load(save_dir+ '/data/onehots_val.npy')
				else:
					onehots_val=0

				#data saved for forward sim
				forwardsim_x_true = np.load(save_dir+ '/data/forwardsim_x_true.npy')
				forwardsim_y = np.load(save_dir+ '/data/forwardsim_y.npy')

				if(use_one_hot):
					forwardsim_onehot = np.load(save_dir+ '/data/forwardsim_onehot.npy')
				else:
					forwardsim_onehot=0

			else:

				######################################
				############ TRAINING DATA ###########
				######################################

				dataX=[]
				dataY=[]
				dataZ=[]
				dataOneHots=[]
				# for rollout_counter in training_rollouts:
				for i in range(training_rollouts/2):

					#read in data from 1 rollout
					# robot_file= data_dir + "/" + str(rollout_counter) + '_robot_info.obj'
					# mocap_file= data_dir + "/" + str(rollout_counter) + '_mocap_info.obj'
					mocap_file = path_lst[2*i]
					robot_file = path_lst[2*i+1]
					robot_info = pickle.load(open(robot_file,'r'))
					mocap_info = pickle.load(open(mocap_file,'r'))

					#turn saved rollout into s
					full_states_for_dataX, actions_for_dataY= rollout_to_states(robot_info, mocap_info, "all")
					abbrev_states_for_dataX, actions_for_dataY = rollout_to_states(robot_info, mocap_info, state_representation)
						#states_for_dataX: (length-1)x24 cuz ignore 1st one (no deriv)
						#actions_for_dataY: (length-1)x2

					#use s to create ds
					states_for_dataZ = full_states_for_dataX[1:,:]-full_states_for_dataX[:-1,:]

					#s,a,ds
					dataX.append(abbrev_states_for_dataX[:-1,:]) #the last one doesnt have a corresponding next state
					dataY.append(actions_for_dataY[:-1,:])
					dataZ.append(states_for_dataZ)

					#create the corresponding one_hot vector
					curr_surface = mocap_file.split("/")[-2].split("_")[0]
					curr_onehot= create_onehot(curr_surface, use_camera, mappings)
					tiled_curr_onehot = np.tile(curr_onehot,(abbrev_states_for_dataX.shape[0]-1,1))
					dataOneHots.append(tiled_curr_onehot)
				
				#save training data
				dataX=np.concatenate(dataX)
				dataY=np.concatenate(dataY)
				dataZ=np.concatenate(dataZ)
				dataOneHots=np.concatenate(dataOneHots)
				np.save(save_dir+ '/data/dataX.npy', dataX)
				np.save(save_dir+'/data/dataY.npy', dataY)
				np.save(save_dir+ '/data/dataZ.npy', dataZ)
				np.save(save_dir+ '/data/dataOneHots.npy', dataOneHots)

				######################################
				########## VALIDATION DATA ###########
				######################################

				states_val = []
				controls_val = []
				onehots_val = []
				# for rollout_counter in validation_rollouts:
				for i in range(validation_rollouts/2):

					#read in data from 1 rollout
					# robot_file= data_dir + "/" + str(rollout_counter) + '_robot_info.obj'
					# mocap_file= data_dir + "/" + str(rollout_counter) + '_mocap_info.obj'
					mocap_file = path_lst[training_rollouts + 2*i]
					robot_file = path_lst[training_rollouts + 2*i+1]
					robot_info = pickle.load(open(robot_file,'r'))
					mocap_info = pickle.load(open(mocap_file,'r'))

					#turn saved rollout into s
					full_states_for_dataX, actions_for_dataY= rollout_to_states(robot_info, mocap_info, "all")
					abbrev_states_for_dataX, actions_for_dataY = rollout_to_states(robot_info, mocap_info, state_representation)
					states_val.append(abbrev_states_for_dataX)
					controls_val.append(actions_for_dataY)

					# Is this data just unlabeled or something????? Why?

					#create the corresponding one_hot vector
					curr_surface = mocap_file.split("/")[-2].split("_")[0]
					curr_onehot= create_onehot(curr_surface, use_camera, mappings)
					tiled_curr_onehot = np.tile(curr_onehot,(abbrev_states_for_dataX.shape[0],1))
					onehots_val.append(tiled_curr_onehot)

				#save validation data
				states_val = np.array(states_val)
				controls_val = np.array(controls_val)
				onehots_val = np.array(onehots_val)
				np.save(save_dir+ '/data/states_val.npy', states_val)
				np.save(save_dir+ '/data/controls_val.npy', controls_val)
				np.save(save_dir+ '/data/onehots_val.npy', onehots_val)

				#set aside un-preprocessed data, to use later for forward sim
				print("inside traindynamics, the dimensions of full state are: ", full_states_for_dataX.shape)
				forwardsim_x_true = full_states_for_dataX[4:16] #use these steps from the last validation rollout
				print(forwardsim_x_true.shape)
				forwardsim_y = actions_for_dataY[4:16] #use these steps from the last validation rollout
				forwardsim_onehot = tiled_curr_onehot[4:16] #use these steps from the last validation rollout

				np.save(save_dir+ '/data/forwardsim_x_true.npy', forwardsim_x_true)
				np.save(save_dir+ '/data/forwardsim_y.npy', forwardsim_y)
				np.save(save_dir+ '/data/forwardsim_onehot.npy', forwardsim_onehot)

			#################################################
			### preprocess the old training dataset
			#################################################

			print("\n#####################################")
			print("Preprocessing 'old' training data")
			print("#####################################\n")
			#every component (i.e. x position) will now be mean 0, std 1

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

			#save mean and std to files for controller to use
			np.save(save_dir+ '/data/mean_x.npy', mean_x)
			np.save(save_dir+ '/data/mean_y.npy', mean_y)
			np.save(save_dir+ '/data/mean_z.npy', mean_z)
			np.save(save_dir+ '/data/std_x.npy', std_x)
			np.save(save_dir+ '/data/std_y.npy', std_y)
			np.save(save_dir+ '/data/std_z.npy', std_z)

			## concatenate state and action, to be used for training dynamics
			inputs = np.concatenate((dataX, dataY), axis=1)
			outputs = np.copy(dataZ)
			onehots = np.copy(dataOneHots)

			#dimensions
			assert inputs.shape[0] == outputs.shape[0]
			numData = inputs.shape[0]
			inputSize = inputs.shape[1]
			outputSize = outputs.shape[1]

			##############################################
			########### THE DYNAMICS MODEL ###############
			##############################################
		
			#which model
			if(use_one_hot):
				if(use_camera):
					from feedforward_network_camera import feedforward_network
				else:
					from feedforward_network_one_hot import feedforward_network
			else:
				from feedforward_network import feedforward_network

			#initialize model
			self.dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, 0, self.x_index, self.y_index, 
								num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, 
								std_x, std_y, std_z, tf_datatype, np_datatype, print_minimal, feedforward_network, 
								use_one_hot, curr_env_onehot, N,one_hot_dims=one_hot_dims)

			#randomly initialize all vars
			sess.run(tf.initialize_all_variables())  ##sess.run(tf.global_variables_initializer()) 

			##############################################
			########## THE AGGREGATION LOOP ##############
			##############################################

			'''TO DO: havent done one-hots for aggregation'''

			counter=0
			training_loss_list=[]
			old_loss_list=[]
			new_loss_list=[]
			dataX_new = np.zeros((0,dataX.shape[1]))
			dataY_new = np.zeros((0,dataY.shape[1]))
			dataZ_new = np.zeros((0,dataZ.shape[1]))
			print("dataX dim: ", dataX.shape)

			if(playback_mode):
				print("making playback controller")
				self.controller = ControllerPlayback(traj_save_path, save_dir, dt_steps, state_representation, desired_shape_for_traj,
									left_min, left_max, right_min, right_max, 
									use_pid_mode=use_pid_mode,
									frequency_value=frequency_value, stateSize=dataX.shape[1], actionSize=dataY.shape[1], 
									N=N, horizon=horizon, serial_port=serial_port, baud_rate=baud_rate, DEFAULT_ADDRS=DEFAULT_ADDRS,visualize_rviz=visualize_rviz)
			else:
				self.controller = Controller(traj_save_path, save_dir, dt_steps, state_representation, desired_shape_for_traj,
									left_min, left_max, right_min, right_max, 
									use_pid_mode=use_pid_mode,
									frequency_value=frequency_value, stateSize=dataX.shape[1], actionSize=dataY.shape[1], 
									N=N, horizon=horizon, serial_port=serial_port, baud_rate=baud_rate, DEFAULT_ADDRS=DEFAULT_ADDRS,visualize_rviz=visualize_rviz)

			saver = tf.train.Saver()
			saver.restore(sess, self.restore_dynamics_model_filepath)

			num_runs_per_setting = 3
			empirical_avg_costs = []
			for i in range(len(max_H_for_K)):
				H = max_H_for_K[i]
				K = K_list[i]

				self.controller.set_N(K)
				self.controller.set_horizon(H)
				
				summed_cost = 0
				for j in range(num_runs_per_setting):
					print
					print
					print("PAUSING... right before a controller run... RESET THE ROBOT TO A GOOD LOCATION BEFORE CONTINUING...")
					print
					print
					IPython.embed()

					# self.controller.start_robot()
					# self.controller.run calls stop_roach in utils.py
					# if i != 0 or j != 0:
					# 	self.controller.setup()
					traj_taken, actions_taken, desired_states = self.controller.run(num_steps_for_rollout=self.num_steps_per_controller_run, aggregation_loop_counter=0, dyn_model=self.dyn_model)
					# self.controller.kill_robot()

					summed_cost += self.compute_cost(traj_taken, desired_states)

				avg_cost = summed_cost/num_runs_per_setting
				empirical_avg_costs.append(avg_cost)

			print("\n\nDONE WITH ALL ROLLOUTS")
			print("killing robot")
			self.controller.kill_robot()
			return empirical_avg_costs
	
	#distance needed for unit 2 to go toward unit1
	#NOT THE CORRECT SIGN
	def moving_distance(self, unit1, unit2):
		phi = (unit2-unit1) % (2*np.pi)

		phi[phi > np.pi] = (2*np.pi-phi)[phi > np.pi]

		return phi

	def compute_cost(self, traj_taken, desired_states):
		# Computes the overall heuristic cost on 1 run
		# traj_taken, as is, is a list of states. [horizon + 1, state size] 
		#  desired_states is a list of arrays
		horiz_penalty_factor = self.controller.horiz_penalty_factor
		backward_discouragement = self.controller.backward_discouragement
		heading_penalty_factor = self.controller.heading_penalty_factor

		traj_taken = np.array(traj_taken)
		resulting_states = np.reshape(traj_taken, (traj_taken.shape[1], 1, traj_taken.shape[0])).T # (horizon + 1, N, state)
		
		N = 1

		curr_line_segment = 0 #Is 0 since traj_taken is the entire trajectory from start to end, as is desired_states
		curr_seg = np.tile(curr_line_segment,(N,))
		curr_seg = curr_seg.astype(int)

		moved_to_next = np.zeros((N,))
		prev_forward = np.zeros((N,))

		scores=np.zeros((N,))
		
		for pt_number in range(resulting_states.shape[0]):
			#array of "the point"... for each sim
			pt = resulting_states[pt_number] # N x state, all states for all N candidates at one time slice

			#arrays of line segment points... for each sim
			curr_start = desired_states[curr_seg]
			curr_end = desired_states[curr_seg+1]
			next_start = desired_states[curr_seg+1]
			next_end = desired_states[curr_seg+2]

			curr_start = np.reshape(curr_start, (1, curr_start.size))
			curr_end = np.reshape(curr_end, (1, curr_end.size))
			next_start = np.reshape(next_start, (1, next_start.size))
			next_end = np.reshape(next_end, (1, next_end.size))

			#vars... for each sim
			min_perp_dist = np.ones((N, ))*5000

			############ closest distance from point to current line segment

			#vars
			# x = pt[:,self.x_index]
			# print(type(x))
			# print(x.shape)
			# y = curr_start[:,0]
			# print(type(y))
			# print(y.shape)
			a = pt[:,self.x_index]- curr_start[:,0]
			b = pt[:,self.y_index]- curr_start[:,1]
			c = curr_end[:,0]- curr_start[:,0]
			d = curr_end[:,1]- curr_start[:,1]

			#project point onto line segment
			which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

			#point on line segment that's closest to the pt
			closest_pt_x = np.copy(which_line_section)
			closest_pt_y = np.copy(which_line_section)
			closest_pt_x[which_line_section<0] = curr_start[:,0][which_line_section<0]
			closest_pt_y[which_line_section<0] = curr_start[:,1][which_line_section<0]
			closest_pt_x[which_line_section>1] = curr_end[:,0][which_line_section>1]
			closest_pt_y[which_line_section>1] = curr_end[:,1][which_line_section>1]
			closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,0] + np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
			closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,1] + np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

			#min dist from pt to that closest point (ie closes dist from pt to line segment)
			min_perp_dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

			#"forward-ness" of the pt... for each sim
			curr_forward = which_line_section

			############ closest distance from point to next line segment

			#vars
			a = pt[:,self.x_index]- next_start[:,0]
			b = pt[:,self.y_index]- next_start[:,1]
			c = next_end[:,0]- next_start[:,0]
			d = next_end[:,1]- next_start[:,1]

			#project point onto line segment
			which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

			#point on line segment that's closest to the pt
			closest_pt_x = np.copy(which_line_section)
			closest_pt_y = np.copy(which_line_section)
			closest_pt_x[which_line_section<0] = next_start[:,0][which_line_section<0]
			closest_pt_y[which_line_section<0] = next_start[:,1][which_line_section<0]
			closest_pt_x[which_line_section>1] = next_end[:,0][which_line_section>1]
			closest_pt_y[which_line_section>1] = next_end[:,1][which_line_section>1]
			closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,0] + np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
			closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,1] + np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

			#min dist from pt to that closest point (ie closes dist from pt to line segment)
			dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

			#pick which line segment it's closest to, and update vars accordingly
			curr_seg[dist<=min_perp_dist] += 1
			moved_to_next[dist<=min_perp_dist] = 1
			curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]#### np.clip(which_line_section,0,1)[dist<=min_perp_dist]
			min_perp_dist = np.min([min_perp_dist, dist], axis=0)

			################## scoring
			#penalize horiz dist
			scores += min_perp_dist*horiz_penalty_factor

			#penalize moving backward
			scores[moved_to_next==0] += (prev_forward - curr_forward)[moved_to_next==0]*backward_discouragement

			#penalize heading away from angle of line
			desired_yaw = np.arctan2(curr_end[:,1]-curr_start[:,1], curr_end[:,0]-curr_start[:,0])
			curr_yaw = np.arctan2(pt[:,self.yaw_sin_index],pt[:,self.yaw_cos_index])
			diff = np.abs(self.moving_distance(desired_yaw, curr_yaw))

			scores += diff*heading_penalty_factor

			#update
			prev_forward = np.copy(curr_forward)
			prev_pt = np.copy(pt)

		# print(scores)
		print(scores[0])
		return scores[0]


def main():
	pass
	# This file is not for running; it just contains a utility function
   
if __name__ == '__main__':
	main()
