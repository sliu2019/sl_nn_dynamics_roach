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

class ControllerPlayback(object):

    def __init__(self, dt_steps, state_representation, min_motor_gain, max_motor_gain, frequency_value=20, stateSize=24, actionSize=2):

      self.desired_shape_for_traj = "zigzag" #straight, left, right, circle_left, zigzag, figure8

      self.frequency_value = frequency_value
      self.state_representation = state_representation

      self.min_motor_gain= min_motor_gain
      self.max_motor_gain= max_motor_gain

      self.mocap_info = PoseStamped()

      #env vars
      self.x_index=0
      self.y_index=1
      self.yaw_cos_index = 10
      self.yaw_sin_index = 11
      self.action_shape = (actionSize,)
      self.min_ac = np.ones(self.action_shape)*self.min_motor_gain
      self.max_ac = np.ones(self.action_shape)*self.max_motor_gain

      self.a = Actions()

      self.dt_steps=dt_steps
      self.stateSize = stateSize
      self.inputSize = self.stateSize + actionSize
      self.outputSize = self.stateSize

      #controller vars
      self.N=1000
      self.horizon = dt_steps*8
      self.horiz_penalty_factor= 20
      self.forward_encouragement_factor= 30
      self.backward_discouragement= 0
      self.heading_penalty_factor= 3
      self.perp_dist_threshold = 0.3

      self.setup()

    def setup(self):

      #init node
      rospy.init_node('controller_playback_node', anonymous=True)

      #make subscribers
      self.sub_playbackState = rospy.Subscriber('/playback_state', Float32MultiArray, self.callback)

      #make publishers
      self.publish_markers= rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
      self.publish_markers_desired= rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)

      #tensorflow options
      gpu_device = 0
      gpu_frac = 0.3
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
      self.config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    def callback(self,info):
      #x,y,z,vx,vy,vz,np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(yw),np.sin(yw),wx,wy,wz,np.cos(al),np.sin(al),np.cos(ar),np.sin(ar),v_al,v_ar,robotinfo.bemfL,robotinfo.bemfR,robotinfo.vBat
      #added taken action as the last 2 entries to this state

      self.curr_state = info.data
      self.got_state = True

    def run(self,num_steps_for_rollout, aggregation_loop_counter, dyn_model):

      #init values for the loop below
      self.got_state = False
      self.dyn_model = dyn_model
      self.traj_taken=[]
      self.actions_taken=[]
      self.curr_line_segment = 0
      take_steps=True
      num_iters=-1
      dt=1
      optimal_action=[self.min_motor_gain,self.min_motor_gain]
      num_paused = 0
      num_unpaused = 0
      run_duration = 10
      pause_duration = 2
      pause=False

      #load in the means
      self.mean_x= np.load('../data/mean_x.npy')
      self.mean_y= np.load('../data/mean_y.npy') 
      self.mean_z= np.load('../data/mean_z.npy') 
      self.std_x= np.load('../data/std_x.npy') 
      self.std_y= np.load('../data/std_y.npy') 
      self.std_z= np.load('../data/std_z.npy') 

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

          while(self.got_state==False):
            junk=1
          print("got state")
          self.got_state= False
          num_iters+=1

          ########################
          #### COMPUTE ACTION ####
          ########################

          if(num_iters==0):
            #create desired trajectory
            print("starting x position: ", self.curr_state[self.x_index])
            print("starting y position: ", self.curr_state[self.y_index])
            self.desired_states = make_trajectory(self.desired_shape_for_traj, self.curr_state, self.x_index, self.y_index)

          if(pause):
            optimal_action=[self.min_motor_gain, self.min_motor_gain]
          else:
            if(num_iters%self.dt_steps == 0):
              self.traj_taken.append(self.curr_state)
              
              optimal_action, curr_line_segment, _, _, _, _, _, _, _ = self.a.compute_optimal_action(np.copy(self.curr_state), sess, self.desired_states, self.min_ac, self.max_ac, np.copy(optimal_action), num_iters, self.dyn_model, self.N, self.horizon, self.dt_steps, self.x_index, self.y_index, self.yaw_cos_index, self.yaw_sin_index, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z, self.publish_markers_desired, self.publish_markers, self.curr_line_segment, self.perp_dist_threshold, self.horiz_penalty_factor, self.backward_discouragement, self.forward_encouragement_factor, self.heading_penalty_factor, 0)
              
              self.curr_line_segment = np.copy(curr_line_segment)
              self.actions_taken.append(optimal_action)

          print("computed action")