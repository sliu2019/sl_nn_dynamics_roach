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
import cv2
from os import system
from scipy.misc import imread


#add nn_dynamics_roach to sys.path
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
                use_pid_mode,
                frequency_value=20, stateSize=24, actionSize=2,
                N=1000, horizon=4, serial_port='/dev/ttyUSB0', camera_serial_port = None, baud_rate = 57600, DEFAULT_ADDRS = ['\x00\x01'],visualize_rviz=False):

      #set vars
      self.visualize_rviz=visualize_rviz
      self.serial_port = serial_port
      self.camera_serial_port = camera_serial_port
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

      '''
      CARPET:
        zigzag: 30, 10, 5 (150, just little to the right of green tape, center height near t)
        right: 40, 10, 5 (120, start just to the left of green horiz tape, near black t)
        left: 30, 10, 5 (150, just little to the right of green tape, up close-ish to green corner)
        straight: 30, 10, 5 (70, start middle of green tape, center height near t)

      GRAVEL:
        zigzag: 
        right: 
        left: 
        straight:

      pics: straight, zigzag, left, right
      STYROFOAM:
        zigzag: 30, 10, 5 (looks good, 160, start just to the right of green horiz tape)
        right: 40, 10, 5 (looks good, 120, start just to the left of green horiz tape, very close to bottom edge)
        left: 30, 10, 5 (looks good, 150, start foot to the right of green horiz tape, a foot away from the 2nd foam)
        straight: 30, 10, 5 (looks good, 70, start in middle of green horiz tape)  '''

      if(self.desired_shape_for_traj=='right'):
        self.horiz_penalty_factor= 75 ## care about staying close to the traj
        self.backward_discouragement= 5  ## care about moving forward
        self.heading_penalty_factor= 5 #2 ## care about turning heading to be same direction as line youre trying to follow (but note that this doesnt bring you closer to the line)
      elif(self.desired_shape_for_traj=='zigzag'):
        self.horiz_penalty_factor= 60 #80 ## care about staying close to the traj
        self.backward_discouragement= 10  ## care about moving forward
        self.heading_penalty_factor= 5
      elif(self.desired_shape_for_traj=='left'):
        self.horiz_penalty_factor= 30 ## care about staying close to the traj
        self.backward_discouragement= 10  ## care about moving forward
        self.heading_penalty_factor= 5
      else:
        self.horiz_penalty_factor= 30 # 70 #care about staying close to the traj
        self.backward_discouragement= 10  ## care about moving forward
        self.heading_penalty_factor= 5 ## care about turning heading to be same direction as line youre trying to follow (but note that this doesnt bring you closer to the line)

      self.setup()

    # Some setters
    def set_N(self, new_N):
      self.N = new_N

    def set_horizon(self, new_horizon):
      self.horizon = new_horizon

    def setup(self):

      #init node
      rospy.init_node('controller_node', anonymous=True)
      
      # rospy.Rate helps keep the frequency of a loop at a fixed value with the help of the sleep function, called at the end of loops
      self.rate = rospy.Rate(self.frequency_value)

      #setup serial, roach bridge, and imu queues
      self.xb, self.robots, shared.imu_queues = setup_roach(self.serial_port, self.baud_rate, self.DEFAULT_ADDRS, self.use_pid_mode, 1)

      #set PID gains
      #IPython.embed()
      for robot in self.robots:
        if(self.use_pid_mode):
          robot.setMotorGains([1800,200,100,0,0, 1800,200,100,0,0])

      #IPython.embed()
      #make subscribers
      self.sub_mocap = rospy.Subscriber('/mocap/pose', PoseStamped, self.callback_mocap)

      #make publishers
      self.publish_robotinfo= rospy.Publisher('/robot0/robotinfo', velroach_msg, queue_size=5)
      self.publish_markers= rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
      self.publish_markers_desired= rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)
      self.pub_full_curr_state= rospy.Publisher('full_curr_state', Float32MultiArray, queue_size=5)

      #IPython.embed()
      #action selector (MPC)
      self.a = Actions(visualize_rviz=self.visualize_rviz)

      #tensorflow options: deprecated, passed straight to dynamics model
      # gpu_device = 0
      # gpu_frac = 0.3
      # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
      # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
      # self.config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    def callback_mocap(self,data):
      self.mocap_info = data

    def kill_robot(self):
      stop_and_exit_roach(self.xb, self.lock, self.robots, self.use_pid_mode)

    def kill_robot_special(self):
      # Prevents sys.exit(1) from being called at end
      stop_and_exit_roach_special(self.xb, self.lock, self.robots, self.use_pid_mode)

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
      list_camera_info = [] #left empty if not using camera

      command_frequency = 0
      time_compute_action = 0
      number_compute_action = 0

      #Preliminary camera "warm-up", since the first open of the camera always has issues
      cap = cv2.VideoCapture(int(self.camera_serial_port[-1]))
      cap.release()

      while True:

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
            if step == 0:
              time_of_last_command = time.time()
            else:
              time_of_current_command = time.time()
              command_frequency += (time_of_current_command - time_of_last_command)
              time_of_last_command = time_of_current_command
            robot.setVelGetTelem(send_action[0], send_action[1])
          else:
            robot.setThrustGetTelem(send_action[0], send_action[1])
        self.lock.release()

        ########################
        #### RECEIVE STATE #####
        ########################

        got_data=False
        start_time = time.time()
        while(got_data==False):
          if (time.time() - start_time)%5 == 0:
            print("Controller is waiting to receive data from robot")
          if (time.time() - start_time) > 15:
            # Unsuccessful run; roach stopped communicating with xbee
            stop_roach(self.lock, self.robots, self.use_pid_mode)
            return None, None, None
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
          #print "got state"

        #collect info to save for later
        list_robot_info.append(robotinfo)
        list_mocap_info.append(self.mocap_info)

        if(step==0):
          old_time= -7
          old_pos= self.mocap_info.pose.position #curr pos
          old_al= robotinfo.posL/math.pow(2,16)*2*math.pi #curr al
          old_ar= robotinfo.posR/math.pow(2,16)*2*math.pi #curr ar

        #check dt of controller
        if(step>0):
          step_dt = (robotinfo.stamp.secs-old_time.secs) + (robotinfo.stamp.nsecs-old_time.nsecs)*0.000000001
          print("DT: ", step_dt)

        #create state from the info
        full_curr_state, _, _, _, _ = singlestep_to_state(robotinfo, self.mocap_info, old_time, old_pos, old_al, old_ar, "all")
        # print("full_curr_state position, after singlesteptostate:", full_curr_state)
        # print("mocap info: ", self.mocap_info)
        abbrev_curr_state, old_time, old_pos, old_al, old_ar = singlestep_to_state(robotinfo, self.mocap_info, old_time, old_pos, old_al, old_ar, self.state_representation)

        # Get live camera input
        if self.camera_serial_port: #If not none  
          cap = cv2.VideoCapture(int(self.camera_serial_port[-1]))
          ret, frame = cap.read()

          #cv2.imshow('frame', frame)

          # A temporary file that it's ok to continuously write over
          # "Image" object in OpenCV to .jpg
          temp_img_filename = "frame.jpg"
          cv2.imwrite(temp_img_filename, frame)
          # Crop to correct size
          system('convert ' + temp_img_filename + ' -crop 480x480+80+0 ' + temp_img_filename + '_cropped.jpg')
          system('convert ' + temp_img_filename + '_cropped.jpg' + ' -resize 227x227 ' + temp_img_filename + '_final.jpg')

          # **********PREPROCESS**************
          # Subtract mean, flip rgb to bgr, then feed into alexnet + random projection + feedforwardnetwork_camera 
          # This should be the mean of the dataset alexnet was trained on....
          
          # REPLACE WITH COMMENTED OUT LINE WHEN TRAINIG WITH REAL LIVE CAMERA
          # training_mean = np.load(self.save_dir + /data/mean_camera.npy')
          # bELOW won't be completely accurate, cause you used each image a different number of times. But at least it's consistent with what you "cheaty-trained' on
          training_mean= [123.68, 116.779, 103.939] 

          img = (imread(temp_img_filename + '_final.jpg')[:,:,:3]).astype(np.float32)

          img = img - training_mean 
          img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]

          # cv2.waitKey(0)
          # cv2.destroyAllWindows()
          cap.release()

          # Add to sequence of images, used for training
          list_camera_info.append(img)
        #########################
        ## CHECK STOPPING COND ##
        #########################

        if(step>num_steps_for_rollout):
          
          print("DONE TAKING ", step, " STEPS.")

          #stop roach
          stop_roach(self.lock, self.robots, self.use_pid_mode)
          # print("after calling stop_roach")
          # IPython.embed()

          #save for playback debugging
          robot_file= self.save_dir +'/'+ self.traj_save_path +'/robot_info.obj'
          mocap_file= self.save_dir +'/'+ self.traj_save_path +'/mocap_info.obj'
          camera_file = self.save_dir +'/'+ self.traj_save_path +'/camera_info.obj'
          pickle.dump(list_robot_info,open(robot_file,'w'))
          pickle.dump(list_mocap_info,open(mocap_file,'w'))
          pickle.dump(list_camera_info,open(camera_file,'w'))

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

          print("Empirical time between commands (in seconds): ", command_frequency/float(num_steps_for_rollout))
          print("Empirical time to execute compute_action for k = ", self.N, " and H = ", self.horizon, " is:", time_compute_action/float(number_compute_action))

          return(self.traj_taken, self.actions_taken, self.desired_states, list_camera_info)

        ########################
        #### COMPUTE ACTION ####
        ########################

        if(step==0):
          #create desired trajectory
          print("starting x position: ", full_curr_state[self.x_index])
          print("starting y position: ", full_curr_state[self.y_index])
          self.desired_states = make_trajectory(self.desired_shape_for_traj, np.copy(full_curr_state), self.x_index, self.y_index)

        if(step%self.dt_steps == 0):
          self.traj_taken.append(full_curr_state)

          time_before = time.time()
          optimal_action, curr_line_segment, old_curr_forward, \
              save_perp_dist, save_forward_dist, saved_old_forward_dist, \
              save_moved_to_next, save_desired_heading, save_curr_heading = self.a.compute_optimal_action(np.copy(full_curr_state), np.copy(abbrev_curr_state), img, self.desired_states, \
                                                                                                          self.left_min, self.left_max, self.right_min, self.right_max, \
                                                                                                          np.copy(optimal_action), step, self.dyn_model, self.N, \
                                                                                                          self.horizon, self.dt_steps, self.x_index, self.y_index, \
                                                                                                          self.yaw_cos_index, self.yaw_sin_index, \
                                                                                                          self.mean_x, self.mean_y, self.mean_z, \
                                                                                                          self.std_x, self.std_y, self.std_z, self.publish_markers_desired, \
                                                                                                          self.publish_markers, self.curr_line_segment, \
                                                                                                          self.horiz_penalty_factor, self.backward_discouragement, \
                                                                                                          self.heading_penalty_factor, self.old_curr_forward)
          time_compute_action += time.time() - time_before
          number_compute_action += 1

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

        #print("computed action")
        self.rate.sleep()
        step+=1

