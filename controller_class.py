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
from threading import Condition
import thread
from Queue import Queue
from collections import OrderedDict
import time, sys, os, traceback
import serial
import math
import pickle

#add nn_dynamics_roach to sys.path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#my imports
import command
import shared_multi as shared
from velociroach import *
from nn_dynamics_roach.msg import velroach_msg
from utils import *
from dynamics_model import Dyn_Model
from trajectories import make_trajectory
from compute_action import Actions


class Controller(object):

    def __init__(self, traj_save_path, save_dir, dt_steps, state_representation, desired_shape_for_traj,
                left_min, left_max, right_min, right_max, 
                use_pid_mode,frequency_value=20, stateSize=24, actionSize=2,
                N=1000, horizon=4, serial_port='/dev/ttyUSB0', baud_rate = 57600, DEFAULT_ADDRS = ['\x00\x01']):

      #set vars
      self.serial_port = serial_port
      self.baud_rate = baud_rate
      self.DEFAULT_ADDRS = DEFAULT_ADDRS
      self.N= N
      self.horizon = horizon
      self.use_pid_mode = use_pid_mode
      self.frequency_value = frequency_value
      self.state_representation = state_representation
      self.desired_shape_for_traj = desired_shape_for_traj
      self.traj_save_path = traj_save_path
      self.save_dir = save_dir
      self.left_min = left_min
      self.left_max = left_max
      self.right_min = right_min
      self.right_max = right_max
      self.action_shape = (actionSize,)
      self.dt_steps=dt_steps
      self.stateSize = stateSize
      self.inputSize = self.stateSize + actionSize
      self.outputSize = self.stateSize

      #read in means and stds
      self.mean_x= np.load(self.save_dir+ '/data/mean_x.npy')
      self.mean_y= np.load(self.save_dir+ '/data/mean_y.npy') 
      self.mean_z= np.load(self.save_dir+ '/data/mean_z.npy') 
      self.std_x= np.load(self.save_dir+ '/data/std_x.npy') 
      self.std_y= np.load(self.save_dir+ '/data/std_y.npy') 
      self.std_z= np.load(self.save_dir+ '/data/std_z.npy') 

      #init vars
      self.lock = Condition()
      self.mocap_info = PoseStamped()

      #env indeces
      self.x_index=0
      self.y_index=1
      self.yaw_cos_index = 10
      self.yaw_sin_index = 11
      
      #FUNCTIONAL on new legs for both straight and circle
      self.horiz_penalty_factor= 40 
      self.backward_discouragement= 10
      self.heading_penalty_factor= 15

      self.setup()

    def setup(self):

      #init node
      rospy.init_node('controller_node', anonymous=True)
      self.rate = rospy.Rate(self.frequency_value)

      #setup serial, roach bridge, and imu queues
      self.xb, self.robots, shared.imu_queues = setup_roach(self.serial_port, self.baud_rate, self.DEFAULT_ADDRS, self.use_pid_mode)

      #set PID gains
      for robot in self.robots:
        if(self.use_pid_mode):
          robot.setMotorGains([1800,200,100,0,0, 1800,200,100,0,0])

      #make subscribers
      self.sub_mocap = rospy.Subscriber('/mocap/pose', PoseStamped, self.callback_mocap)

      #make publishers
      self.publish_robotinfo= rospy.Publisher('/robot0/robotinfo', velroach_msg, queue_size=5)
      self.publish_markers= rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
      self.publish_markers_desired= rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)
      self.pub_curr_state= rospy.Publisher('curr_state', Float32MultiArray, queue_size=5)

      #action selector (MPC)
      self.a = Actions()

      #tensorflow options
      gpu_device = 0
      gpu_frac = 0.3
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
      self.config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    def callback_mocap(self,data):
      self.mocap_info = data

    def kill_robot(self):
      stop_and_exit_roach(self.xb, self.lock, self.robots, self.use_pid_mode)

    def run(self,num_steps_for_rollout, aggregation_loop_counter, dyn_model):

      #init values for the loop below
      self.dyn_model = dyn_model
      self.traj_taken=[]
      self.actions_taken=[]
      self.save_perp_dist=[]
      self.save_forward_dist=[]
      self.saved_old_forward_dist=[]
      self.save_moved_to_next=[]
      self.save_desired_heading=[]
      self.save_curr_heading=[]
      self.curr_line_segment = 0
      self.old_curr_forward=0
      step=0
      optimal_action=[0,0]

      list_robot_info=[]
      list_mocap_info=[]

      while(True):

        if(step%10==0):
          print "     step #: ", step

        ########################
        ##### SEND COMMAND #####
        ########################

        self.lock.acquire()
        for robot in self.robots:
          ##send_action = [0,0]
          send_action = np.copy(optimal_action)
          print "\nsent action: ", send_action[0], send_action[1]
          if(self.use_pid_mode):
            robot.setVelGetTelem(send_action[0], send_action[1])
          else:
            robot.setThrustGetTelem(send_action[0], send_action[1])
        self.lock.release()

        ########################
        #### RECEIVE STATE #####
        ########################

        got_data=False
        while(got_data==False):
          for q in shared.imu_queues.values():
            #while loop, because sometimes, you get multiple things from robot
            #but they're all same, so just use the last one
            while not q.empty():
              d = q.get()
              got_data=True

        if(got_data):
          robotinfo=velroach_msg()
          robotinfo.stamp = rospy.Time.now()
          robotinfo.curLeft = optimal_action[0]
          robotinfo.curRight = optimal_action[1]
          robotinfo.posL = d[2]
          robotinfo.posR = d[3]
          robotinfo.gyroX = d[8]
          robotinfo.gyroY = d[9]
          robotinfo.gyroZ = d[10]
          robotinfo.bemfL = d[14]
          robotinfo.bemfR = d[15]
          robotinfo.vBat = d[16]
          self.publish_robotinfo.publish(robotinfo)
          print "got state"

        #collect info to save for later
        list_robot_info.append(robotinfo)
        list_mocap_info.append(self.mocap_info)

        if(step==0):
          old_time= -7
          old_pos= self.mocap_info.pose.position #curr pos
          old_al= robotinfo.posL/math.pow(2,16)*2*math.pi #curr al
          old_ar= robotinfo.posR/math.pow(2,16)*2*math.pi #curr ar

        #create state from the info
        curr_state, old_time, old_pos, old_al, old_ar = singlestep_to_state(robotinfo, self.mocap_info, old_time, old_pos, old_al, old_ar, self.state_representation)

        #########################
        ## CHECK STOPPING COND ##
        #########################

        if(step>num_steps_for_rollout):
          
          print("DONE TAKING ", step, " STEPS.")

          #stop roach
          stop_roach(self.lock, self.robots, self.use_pid_mode)
          
          #save for playback debugging
          robot_file= self.save_dir +'/'+ self.traj_save_path +'/robot_info.obj'
          mocap_file= self.save_dir +'/'+ self.traj_save_path +'/mocap_info.obj'
          pickle.dump(list_robot_info,open(robot_file,'w'))
          pickle.dump(list_mocap_info,open(mocap_file,'w'))

          #save
          np.save(self.save_dir +'/'+ self.traj_save_path +'/actions.npy', self.actions_taken)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/desired.npy', self.desired_states)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/executed.npy', self.traj_taken)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/perp.npy', self.save_perp_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/forward.npy', self.save_forward_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/oldforward.npy', self.saved_old_forward_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/movedtonext.npy', self.save_moved_to_next)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/desheading.npy', self.save_desired_heading)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/currheading.npy', self.save_curr_heading)

          return(self.traj_taken, self.actions_taken, self.desired_states)

        ########################
        #### COMPUTE ACTION ####
        ########################

        if(step==0):
          #create desired trajectory
          print("starting x position: ", curr_state[self.x_index])
          print("starting y position: ", curr_state[self.y_index])
          self.desired_states = make_trajectory(self.desired_shape_for_traj, np.copy(curr_state), self.x_index, self.y_index)

        if(step%self.dt_steps == 0):
          self.traj_taken.append(curr_state)
          optimal_action, curr_line_segment, old_curr_forward, save_perp_dist, save_forward_dist, saved_old_forward_dist, save_moved_to_next, save_desired_heading, save_curr_heading = self.a.compute_optimal_action(np.copy(curr_state), self.desired_states, self.left_min, self.left_max, self.right_min, self.right_max, np.copy(optimal_action), step, self.dyn_model, self.N, self.horizon, self.dt_steps, self.x_index, self.y_index, self.yaw_cos_index, self.yaw_sin_index, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z, self.publish_markers_desired, self.publish_markers, self.curr_line_segment, self.horiz_penalty_factor, self.backward_discouragement, self.heading_penalty_factor, self.old_curr_forward)
          self.curr_line_segment = np.copy(curr_line_segment)
          self.old_curr_forward = np.copy(old_curr_forward)

          #if(step>(num_steps_for_rollout-2)):
          #  optimal_action=[0,0]

          self.actions_taken.append(optimal_action)
          self.save_perp_dist.append(save_perp_dist)
          self.save_forward_dist.append(save_forward_dist)
          self.saved_old_forward_dist.append(saved_old_forward_dist)
          self.save_moved_to_next.append(save_moved_to_next)
          self.save_desired_heading.append(save_desired_heading)
          self.save_curr_heading.append(save_curr_heading)

        print("computed action")
        self.rate.sleep()
        step+=1