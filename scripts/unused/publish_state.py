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
from std_msgs.msg import Float32MultiArray


def run():
	
	#init vars
	rollout_num= sys.argv[1]

	#read in a rollout
	states= np.load('../data/'+str(rollout_num)+'_states.npy') #x,y,z,r,p,yw,wx,wy,wz,radians_left,radians_right, backemf_l, backemf_r, vbat
	actions= np.load('../data/'+str(rollout_num)+'_actions.npy')
	dt_vals= np.load('../data/'+str(rollout_num)+'_dt.npy') #curr - past

	#states= np.load('debug_states.npy') #x,y,z,r,p,yw,wx,wy,wz,radians_left,radians_right, backemf_l, backemf_r, vbat
	#dt_vals= np.load('debug_dt.npy')

	#imp/change
	state_representation="all"
	frequency_value = 10

	#don't change
	rospy.init_node('controller_node', anonymous=True)
	rate = rospy.Rate(frequency_value)
	publish_state= rospy.Publisher('/playback_state', Float32MultiArray, queue_size=5)
	counter=0

	#pick out the cols
	x = states[1:, 0]
	y = states[1:, 1]
	z = states[1:, 2]
	r = states[1:, 3]
	p = states[1:, 4]
	yw = states[1:, 5]
	wx = states[1:, 6]
	wy = states[1:, 7]
	wz = states[1:, 8]
	al = states[1:, 9]
	ar = states[1:, 10]
	bemf_l = states[1:, 11]
	bemf_r = states[1:, 12]
	vbat = states[1:, 13]

	#convert r,p,y to rad
	r=r*np.pi/180.0
	p=p*np.pi/180.0
	yw=yw*np.pi/180.0

	#calculate com vel
	smooth_x = states[:,0]
	smooth_y = states[:,1]
	smooth_z = states[:,2]
	vel_x = (smooth_x[1:]-smooth_x[:-1])/dt_vals[1:]
	vel_y = (smooth_y[1:]-smooth_y[:-1])/dt_vals[1:]
	vel_z = (smooth_z[1:]-smooth_z[:-1])/dt_vals[1:]

	#calculate motor vel
	smooth_al = states[:,9]
	smooth_ar = states[:,10]
	vel_al = (smooth_al[1:]-smooth_al[:-1])/dt_vals[1:]
	vel_ar = (smooth_ar[1:]-smooth_ar[:-1])/dt_vals[1:]

	#create the state
	if(state_representation=="all"):
		#all... x y z vx vy vz cos(r) sin(r) cos(p) sin(p) cos(yw) sin(yw) wx wy wz cos(al) sin(al) cos(ar) sin(ar) v_al v_ar bemfl bemfr vbat
		all_states = np.array([x, y, z, vel_x, vel_y, vel_z, np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), wx, wy, wz, np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), vel_al, vel_ar, bemf_l, bemf_r, vbat])

	if(state_representation=="only_yaw"):
		#only yaw
		all_states = np.array([np.cos(states[:-1,5]), np.sin(states[:-1,5])])
	if(state_representation=="no_legs"):
		#no leg position
		all_states = np.array([states[:-1,0], states[:-1,1], states[:-1,2], vel_x, vel_y, vel_z, np.cos(states[:-1,3]), np.sin(states[:-1,3]), np.cos(states[:-1,4]), np.sin(states[:-1,4]), np.cos(states[:-1,5]), np.sin(states[:-1,5]), states[:-1,6], states[:-1,7], states[:-1,8], vel_al, vel_ar, states[:-1,11], states[:-1,12], states[:-1,13]])
	if(state_representation=="no_legs_gyro"):
		#no leg position + no gyro
		all_states = np.array([states[:-1,0], states[:-1,1], states[:-1,2], vel_x, vel_y, vel_z, np.cos(states[:-1,3]), np.sin(states[:-1,3]), np.cos(states[:-1,4]), np.sin(states[:-1,4]), np.cos(states[:-1,5]), np.sin(states[:-1,5]), vel_al, vel_ar, states[:-1,11], states[:-1,12], states[:-1,13]])
	if(state_representation=="only_mocap"):
		#only mocap stuff... x y z vx vy vz cos(r) sin(r) cos(p) sin(p) cos(yw) sin(yw)
		all_states = np.array([states[:-1,0], states[:-1,1], states[:-1,2], vel_x, vel_y, vel_z, np.cos(states[:-1,3]), np.sin(states[:-1,3]), np.cos(states[:-1,4]), np.sin(states[:-1,4]), np.cos(states[:-1,5]), np.sin(states[:-1,5])])

	all_states = np.transpose(all_states)


	####################################
	########### MAIN LOOP ##############
	####################################
	
	while not rospy.is_shutdown():

		my_message = Float32MultiArray()
		my_message.data= np.copy(all_states[counter,:])
		publish_state.publish(my_message)

		counter+=1
		rate.sleep()

if __name__ == '__main__':
	try:
		run()
	except:
		print('************************')
		print("error--- ", sys.exc_info())