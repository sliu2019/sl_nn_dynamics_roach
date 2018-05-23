#!/usr/bin/env python

import math
import rospy
import numpy as np
import numpy.random as npr
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import copy 
import sys
import tensorflow as tf
import signal
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time
import IPython
import matplotlib.pyplot as plt
import pickle

#mine
from dynamics_model import Dyn_Model
from trajectories import make_trajectory
from dynamics_model import Dyn_Model
from compute_action import Actions

#others
from threading import Condition
import thread
from Queue import Queue
from collections import OrderedDict

# Roach Imports
import command
import time, sys, os, traceback
import serial
from velociroach import *
from roach_dynamics_learning.msg import velroach_msg
import shared_multi as shared
import math
from utils import *

class ControllerPlayback(object):

    # def __init__(self, save_dir, dt_steps, state_representation, min_motor_gain, max_motor_gain, frequency_value=20, stateSize=24, actionSize=2):
    def __init__(self, traj_save_path, save_dir, dt_steps, state_representation, desired_shape_for_traj,
                left_min, left_max, right_min, right_max, 
                use_pid_mode,
                frequency_value=20, stateSize=24, actionSize=2,
                N=1000, horizon=4, serial_port='/dev/ttyUSB0', baud_rate = 57600, DEFAULT_ADDRS = ['\x00\x01'],visualize_rviz=False):

      self.desired_shape_for_traj = desired_shape_for_traj
      self.visualize_rviz = visualize_rviz
      self.frequency_value = frequency_value
      self.state_representation = state_representation

      # self.min_motor_gain= min_motor_gain
      # self.max_motor_gain= max_motor_gain

      self.mocap_info = PoseStamped()

      self.save_dir = save_dir
      self.left_min = left_min
      self.left_max = left_max
      self.right_min = right_min
      self.right_max = right_max

      #env vars
      self.x_index=0
      self.y_index=1
      self.yaw_cos_index = 10
      self.yaw_sin_index = 11
      self.action_shape = (actionSize,)
      # self.min_ac = np.ones(self.action_shape)*self.min_motor_gain
      # self.max_ac = np.ones(self.action_shape)*self.max_motor_gain

      self.a = Actions(visualize_rviz=self.visualize_rviz)

      self.dt_steps=dt_steps
      self.stateSize = stateSize
      self.inputSize = self.stateSize + actionSize
      self.outputSize = self.stateSize

      #controller vars
      self.N= N
      self.horizon = horizon
      self.horiz_penalty_factor= 20
      self.backward_discouragement= 0
      self.heading_penalty_factor= 3

      #read in means and stds
      self.mean_x= np.load(self.save_dir+ '/data/mean_x.npy')
      self.mean_y= np.load(self.save_dir+ '/data/mean_y.npy') 
      self.mean_z= np.load(self.save_dir+ '/data/mean_z.npy') 
      self.std_x= np.load(self.save_dir+ '/data/std_x.npy') 
      self.std_y= np.load(self.save_dir+ '/data/std_y.npy') 
      self.std_z= np.load(self.save_dir+ '/data/std_z.npy')

      self.setup()

    def setup(self):

      #init node
      rospy.init_node('controller_playback_node', anonymous=True)

      #make publishers
      self.publish_markers= rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
      self.publish_markers_desired= rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)

      #tensorflow options
      gpu_device = 0
      gpu_frac = 0.3
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
      self.config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    def run(self,num_steps_for_rollout, aggregation_loop_counter, dyn_model):

      #init values for the loop below
      self.dyn_model = dyn_model
      self.actions_taken=[]
      self.save_perp_dist=[]
      self.save_forward_dist=[]
      self.saved_old_forward_dist=[]
      self.save_moved_to_next=[]
      self.save_desired_heading=[]
      self.save_curr_heading=[]
      self.curr_line_segment = 0
      self.old_curr_forward=0
      take_steps=True
      num_iters=0
      dt=1
      optimal_action=[0, 0]
      num_paused = 0
      num_unpaused = 0
      run_duration = 10
      pause_duration = 2
      pause=False

      # task_type=['carpet']
      # data_path = os.path.abspath(os.path.join(os.getcwd(), "../data_collection/"))
      # path_lst = []

      # for subdir, dirs, files in os.walk(data_path):
      #     lst = subdir.split("/")[-1].split("_")
      #     if len(lst) >= 3:
      #         surface = lst[0]
      #         month = lst[2]
      #         if surface in task_type or task_type == "all" and month in months:
      #             for file in files:
      #                 path_lst.append(os.path.join(subdir, file))
      # path_lst.sort()
      # path_lst = path_lst[360:]

      # all_states = []
      # all_robot_info = []
      # all_mocap_info = []

      # for i in range(len(path_lst)/2):

      #   mocap_file = path_lst[2*i]
      #   robot_file = path_lst[2*i+1]
      #   robot_info = pickle.load(open(robot_file,'r'))
      #   mocap_info = pickle.load(open(mocap_file,'r'))

      #   all_robot_info.extend(robot_info)
      #   all_mocap_info.extend(mocap_info)

      #   #turn saved rollout into s
      #   states, _= rollout_to_states(robot_info, mocap_info, self.state_representation) 
      #   all_states.extend(states)

      #pretty straight
      robot_file = "../data_collection/carpet_2018_02_13_11_52_07/0_robot_info.obj"
      mocap_file = "../data_collection/carpet_2018_02_13_11_52_07/0_mocap_info.obj"

      #kind of curvy
      robot_file = "../data_collection/carpet_2018_02_13_11_44_24/9_robot_info.obj"
      mocap_file = "../data_collection/carpet_2018_02_13_11_44_24/9_mocap_info.obj"

      #turn
      '''robot_file = "../data_collection/carpet_2018_02_13_11_44_24/4_robot_info.obj"
      mocap_file = "../data_collection/carpet_2018_02_13_11_44_24/4_mocap_info.obj"

      #gravel
      robot_file = "../data_collection/gravel_2018_02_02_18_07_34/0_robot_info.obj"
      mocap_file = "../data_collection/gravel_2018_02_02_18_07_34/0_mocap_info.obj"

      #styrofoam
      robot_file = "../data_collection/styrofoam_2018_02_16_15_05_52/0_robot_info.obj"
      mocap_file = "../data_collection/styrofoam_2018_02_16_15_05_52/0_mocap_info.obj"   '''   


      all_robot_info = pickle.load(open(robot_file,'r'))
      all_mocap_info = pickle.load(open(mocap_file,'r'))

      states, _ = rollout_to_states(all_robot_info, all_mocap_info, self.state_representation)

      with tf.Session(config=self.config) as sess:
        
        while(take_steps==True):

          if(num_iters%10==0):
            print "\n", "****** step #: ", num_iters

          #pause execution every once in a while
          '''if(num_unpaused<run_duration):
            num_paused=0
            num_unpaused+=1
            pause = False
          else:
            if(num_paused==pause_duration):
              num_unpaused=0
              pause=True
            if(num_paused<pause_duration):
              num_paused+=1
              pause=True'''

          ########################
          ##### SEND COMMAND #####
          ########################

          send_action = np.copy(optimal_action)
          print "\nsent action: ", send_action[0], send_action[1]

          ########################
          #### RECEIVE STATE #####
          ########################
          robotinfo = all_robot_info[num_iters]
          mocapinfo = all_mocap_info[num_iters]

          if(num_iters==0):
            old_time= -7
            old_pos= self.mocap_info.pose.position #curr pos
            old_al= robotinfo.posL/math.pow(2,16)*2*math.pi #curr al
            old_ar= robotinfo.posR/math.pow(2,16)*2*math.pi #curr ar

          #check dt of controller
          if(num_iters>0):
            step_dt = (robotinfo.stamp.secs-old_time.secs) + (robotinfo.stamp.nsecs-old_time.nsecs)*0.000000001
            print("DT: ", step_dt)

          curr_state, old_time, old_pos, old_al, old_ar = singlestep_to_state(robotinfo, mocapinfo, old_time, old_pos, old_al, old_ar, self.state_representation)

          ########################
          #### COMPUTE ACTION ####
          ########################

          if(num_iters==0):
            #create desired trajectory
            print("starting x position: ", curr_state[self.x_index])
            print("starting y position: ", curr_state[self.y_index])
            
            ##predict actions to make you follow the specified traj
            #self.desired_states = make_trajectory(self.desired_shape_for_traj, curr_state, self.x_index, self.y_index)

          ##predict actions to keep you along the true executed trajectory
          self.desired_states = states[num_iters:num_iters+15]

          if(num_iters%self.dt_steps == 0):
            
            optimal_action, curr_line_segment, old_curr_forward, save_perp_dist, save_forward_dist, saved_old_forward_dist, save_moved_to_next, save_desired_heading, save_curr_heading = self.a.compute_optimal_action(np.copy(curr_state), self.desired_states, self.left_min, self.left_max, self.right_min, self.right_max, np.copy(optimal_action), num_iters, self.dyn_model, self.N, self.horizon, self.dt_steps, self.x_index, self.y_index, self.yaw_cos_index, self.yaw_sin_index, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z, self.publish_markers_desired, self.publish_markers, self.curr_line_segment, self.horiz_penalty_factor, self.backward_discouragement, self.heading_penalty_factor, self.old_curr_forward)
            
            self.curr_line_segment = np.copy(curr_line_segment)
            self.old_curr_forward = np.copy(old_curr_forward)
            self.actions_taken.append(optimal_action)
            self.save_perp_dist.append(save_perp_dist)
            self.save_forward_dist.append(save_forward_dist)
            self.saved_old_forward_dist.append(saved_old_forward_dist)
            self.save_moved_to_next.append(save_moved_to_next)
            self.save_desired_heading.append(save_desired_heading)
            self.save_curr_heading.append(save_curr_heading)

          #print("computed action")
          num_iters+=1