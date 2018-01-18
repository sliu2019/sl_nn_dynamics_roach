#!/usr/bin/env python

import math
import rospy
import numpy as np
import numpy.random as npr
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
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

###############################
####### VARS TO SPECIFY #######
###############################

task_type='carpet'
num_rollouts = 10

use_pid_mode = False
slow_pid_mode = False
use_joystick= False
print_frequency = 10

serial_port = '/dev/ttyUSB1'
baud_rate = 57600
DEFAULT_ADDRS = ['\x00\x01']
frequency_value = 10
rollout_length= 50

#room dimensions
  # x in (-1, 2), y in (-1.25, 1.75)
center = [0.50, 0.25]
radius = 1.50

  # x in (-1.4,2.5)
  # y in (-1.2, 1.8)

###############################
######## MOTOR LIMITS #########
###############################

MIN_LEFT = 800
MIN_RIGHT = 800
MAX_LEFT = 2600
MAX_RIGHT = 2600

if(use_pid_mode):
  if(slow_pid_mode):
    MIN_LEFT = 2*math.pow(2,16)*0.001
    MIN_RIGHT = 2*math.pow(2,16)*0.001 
    MAX_LEFT = 6*math.pow(2,16)*0.001
    MAX_RIGHT = 6*math.pow(2,16)*0.001
  else:
    MIN_LEFT = 4*math.pow(2,16)*0.001
    MIN_RIGHT = 4*math.pow(2,16)*0.001
    MAX_LEFT = 12*math.pow(2,16)*0.001
    MAX_RIGHT = 12*math.pow(2,16)*0.001


###############################
######## HELPER FUNCS #########
###############################

#callback for mocap info
def callback_mocap(data):
  global mocap_info
  mocap_info = data

#callback for joystick
def callback_joystick(command):
  if(use_joystick):
    global command_from_joystick
    global lock

    lock.acquire()
    command_from_joystick = convert_command(command)
    lock.release()
  else:
    junk=1

#convert joystick command into motor command
def convert_command(input_val):
  l = input_val.linear.x
  r = input_val.angular.z

  value0= MIN_LEFT + (MAX_LEFT - MIN_LEFT) *l
  value1= MIN_RIGHT + (MAX_RIGHT - MIN_RIGHT)* r

  return [value0, value1, input_val.linear.y == 1 or input_val.linear.z == 1 or input_val.angular.x == 1, input_val.angular.y == 1]

###############################
######## INITIALIZE ###########
###############################

#init ROS node
rospy.init_node('data_collection', anonymous=True)
rate = rospy.Rate(frequency_value)
counter_turn=0

#setup serial, roach bridge, and imu queues
xb, robots, shared.imu_queues = setup_roach(serial_port, baud_rate, DEFAULT_ADDRS, use_pid_mode)

#set PID gains
for robot in robots:
  if(use_pid_mode):
    robot.setMotorGains([1800,200,100,0,0, 1800,200,100,0,0])

#setup ROS subscribers
sub_joystick = rospy.Subscriber('/robot0/cmd_vel', Twist, callback_joystick) #joystick values, published by mjo.py
sub_mocap = rospy.Subscriber('/mocap/pose', PoseStamped, callback_mocap) #mocap data, published by mocap.launch

#setup ROS publishers
publish_robotinfo= rospy.Publisher('/robot0/robotinfo', velroach_msg, queue_size=5) #publish robotinfo from roach

#init vars
lock = Condition()
mocap_info = PoseStamped()
command_from_joystick=[0,0]

#directory for saving collected data
from datetime import datetime
exp_name = task_type + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_dir = os.path.join(os.path.join(os.getcwd()), "..", "data_collection", exp_name)
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

###############################
###### PERFORM ROLLOUT ########
###############################

def run(run_num):
  global lock
  global counter_turn
  start_roach(xb, lock, robots, use_pid_mode)

  #init values for the loop below
  step=0
  selected_action=[0,0]
  list_robot_info=[]
  list_mocap_info=[]

  while(step<rollout_length):

    if(step%print_frequency==0):
      shouldPrint=True
      print "\n", "    step ", step
    else:
      shouldPrint=False

    ########################
    ##### SEND COMMAND #####
    ########################

    lock.acquire()
    for robot in robots:

      #select action to send
      ##send_action = [0, 0]
      send_action = np.copy(selected_action)
      if(shouldPrint):
        print "    sent action: ", send_action[0], send_action[1]

      #send either direct thrust, or target velocity
      if(use_pid_mode):
        robot.setVelGetTelem(send_action[0], send_action[1]) 
      else:
        robot.setThrustGetTelem(send_action[0], send_action[1]) 
    lock.release()

    ##############################################
    #### RECEIVE/PUBLISH INFO FROM ROACH #########
    ##############################################

    got_data=False
    while(got_data==False):
      for q in shared.imu_queues.values():
        #while loop, because sometimes, you get multiple things from robot
        #but they're all same, so just use the last one
        while not q.empty():
          d = q.get()
          '''this used to be encL, encR, gyroX, gyroY, gyroZ, bemfL, bemfR, Vbatt... TO DO: check what this is + fix this comment'''
          got_data=True

    if(got_data):
      robot_info = velroach_msg()
      robot_info.stamp = rospy.Time.now()
      robot_info.curLeft = selected_action[0]
      robot_info.curRight = selected_action[1]
      robot_info.posL = d[2]
      robot_info.posR = d[3]
      robot_info.gyroX = d[8]
      robot_info.gyroY = d[9]
      robot_info.gyroZ = d[10]
      robot_info.bemfL = d[14]
      robot_info.bemfR = d[15]
      robot_info.vBat = d[16]
      publish_robotinfo.publish(robot_info)
      if(shouldPrint):
        print "    got state"

    list_robot_info.append(robot_info)
    list_mocap_info.append(mocap_info)

    ########################
    #### COMPUTE ACTION ####
    ########################

    if(use_joystick==False):
      xpos = mocap_info.pose.position.x
      ypos = mocap_info.pose.position.y
      distance = np.sqrt((xpos-center[0])**2 + (ypos-center[1])**2)

      #random action
      if (distance <= radius):
        selected_action[0] = npr.uniform(MIN_LEFT, MAX_LEFT)
        selected_action[1] = npr.uniform(MIN_RIGHT, MAX_RIGHT)
      #force a turn to stay in region
      else:
        if(counter_turn%2==0):
          print("TURN LEFT")
          # turn LEFT
          selected_action[0] = npr.uniform(MIN_LEFT, MIN_LEFT+(MAX_LEFT-MIN_LEFT)/3.0)
          selected_action[1] = npr.uniform(MAX_RIGHT-(MAX_RIGHT-MIN_RIGHT)/3.0, MAX_RIGHT)
        else:
          print("TURN RIGHT")
          # turn RIGHT
          selected_action[1] = npr.uniform(MIN_RIGHT, MIN_RIGHT+(MAX_RIGHT-MIN_RIGHT)/3.0)
          selected_action[0] = npr.uniform(MAX_LEFT-(MAX_LEFT-MIN_LEFT)/3.0, MAX_LEFT)

    else:
      selected_action = command_from_joystick
        
    #wait for some time
    rate.sleep()
    step+=1

  ########################
  ##### SAVE ROLLOUT #####
  ########################

  robot_file=data_dir + "/" + str(run_num) + '_robot_info.obj'
  mocap_file=data_dir + "/" + str(run_num) + '_mocap_info.obj'

  pickle.dump(list_robot_info,open(robot_file,'w')) 
  pickle.dump(list_mocap_info,open(mocap_file,'w'))

  ########################
  ###### STOP MOVING #####
  ########################

  counter_turn+=1
  print 'DONE WITH ROLLOUT ', run_num, '\n\n'
  stop_roach(lock, robots, use_pid_mode)

##########################
#### COLLECT ROLLOUTS ####
##########################

if __name__ == '__main__':
  try:
    j = int(sys.argv[1])
    for run_num in range(j, j + num_rollouts):
      print "******** rollout # ", run_num
      run(run_num)
      time.sleep(1)

    print('Stopping robot and exiting...')
    stop_and_exit_roach(xb, lock, robots, use_pid_mode)

  except:
    print('************************')
    print("ERROR--- ", sys.exc_info())