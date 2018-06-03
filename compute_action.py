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
import time, sys, os, traceback
import serial

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

class Actions(object):

    def __init__(self,visualize_rviz=False):
      junk=1
      self.visualize_rviz = visualize_rviz


    def compute_optimal_action(self,full_curr_state, abbrev_curr_state, desired_states, left_min, left_max, right_min, right_max, currently_executing_action, step, dyn_model, N, horizon, dt_steps, x_index, y_index, yaw_cos_index, yaw_sin_index, mean_x, mean_y, mean_z, std_x, std_y, std_z, publish_markers_desired, publish_markers, curr_line_segment, horiz_penalty_factor, backward_discouragement, heading_penalty_factor, old_curr_forward):

      #check if curr point in closest to curr_line_segment or if it moved on to next one
      curr_start = desired_states[curr_line_segment]
      curr_end = desired_states[curr_line_segment+1]
      next_start = desired_states[curr_line_segment+1]
      next_end = desired_states[curr_line_segment+2]
      ############ closest distance from point to current line segment
      #vars
      a = full_curr_state[x_index]- curr_start[0]
      b = full_curr_state[y_index]- curr_start[1]
      c = curr_end[0]- curr_start[0]
      d = curr_end[1]- curr_start[1]
      #project point onto line segment
      which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))
      #point on line segment that's closest to the pt
      if(which_line_section<0):
        closest_pt_x = curr_start[0]
        closest_pt_y = curr_start[1]
      elif(which_line_section>1):
        closest_pt_x = curr_end[0]
        closest_pt_y = curr_end[1]
      else:
        closest_pt_x= curr_start[0] + np.multiply(which_line_section,c)
        closest_pt_y= curr_start[1] + np.multiply(which_line_section,d)
      #min dist from pt to that closest point (ie closes dist from pt to line segment)
      min_perp_dist = np.sqrt((full_curr_state[x_index]-closest_pt_x)*(full_curr_state[x_index]-closest_pt_x) + (full_curr_state[y_index]-closest_pt_y)*(full_curr_state[y_index]-closest_pt_y))
      #"forward-ness" of the pt... for each sim
      curr_forward = which_line_section
      ############ closest distance from point to next line segment
      #vars
      a = full_curr_state[x_index]- next_start[0]
      b = full_curr_state[y_index]- next_start[1]
      c = next_end[0]- next_start[0]
      d = next_end[1]- next_start[1]
      #project point onto line segment
      which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))
      #point on line segment that's closest to the pt
      if(which_line_section<0):
        closest_pt_x = next_start[0]
        closest_pt_y = next_start[1]
      elif(which_line_section>1):
        closest_pt_x = next_end[0]
        closest_pt_y = next_end[1]
      else:
        closest_pt_x= next_start[0] + np.multiply(which_line_section,c)
        closest_pt_y= next_start[1] + np.multiply(which_line_section,d)
      #min dist from pt to that closest point (ie closes dist from pt to line segment)
      dist = np.sqrt((full_curr_state[x_index]-closest_pt_x)*(full_curr_state[x_index]-closest_pt_x) + (full_curr_state[y_index]-closest_pt_y)*(full_curr_state[y_index]-closest_pt_y))
      #pick which line segment it's closest to, and update vars accordingly
      moved_to_next = False
      if(dist<min_perp_dist):
        print(" **************************** MOVED ONTO NEXT LINE SEG")
        curr_line_segment+=1
        curr_forward= which_line_section
        min_perp_dist = np.copy(dist)
        moved_to_next = True
      #headings
      curr_start = desired_states[curr_line_segment]
      curr_end = desired_states[curr_line_segment+1]
      desired_yaw = np.arctan2(curr_end[1]-curr_start[1], curr_end[0]-curr_start[0])
      curr_yaw = np.arctan2(full_curr_state[yaw_sin_index],full_curr_state[yaw_cos_index])

      save_perp_dist = np.copy(min_perp_dist)
      save_forward_dist = np.copy(curr_forward)
      saved_old_forward_dist = np.copy(old_curr_forward)
      save_moved_to_next = np.copy(moved_to_next)
      save_desired_heading = np.copy(desired_yaw)
      save_curr_heading = np.copy(curr_yaw)

      old_curr_forward = np.copy(curr_forward)

      ####################################################

      #randomly sample actions to try (the 1st step for each is going to be the currently executing action)
      all_samples = npr.uniform([left_min, right_min], [left_max, right_max], (N, horizon+1, 2))

      #repeat acs for dt_steps each
      for temp in range(all_samples.shape[1]):
        if(temp%dt_steps==0):
          temp_counter=0
          while(temp_counter<dt_steps):
            if((temp+temp_counter)<all_samples.shape[1]):
              all_samples[:,temp+temp_counter,:]=np.copy(all_samples[:,temp,:])
            temp_counter+=1

      #make the 1st one be the currently executing action
      for first_counter in range(dt_steps):
        all_samples[:,first_counter,0]=currently_executing_action[0]
        all_samples[:,first_counter,1]=currently_executing_action[1]

      #run forward sim to predict possible trajectories
      many_in_parallel=True
      resulting_states = dyn_model.do_forward_sim([full_curr_state,0], np.copy(all_samples), None, many_in_parallel, None, None)
      resulting_states= np.array(resulting_states) #this is [horizon+1, N, statesize]
      #print("shape of resulting_states: ", resulting_states.shape)

      #evaluate the trajectories
      scores=np.zeros((N,))
      done_forever=np.zeros((N,))
      move_to_next=np.zeros((N,))
      curr_seg = np.tile(curr_line_segment,(N,))
      curr_seg = curr_seg.astype(int)
      prev_forward = np.zeros((N,))
      moved_to_next = np.zeros((N,))

      prev_pt = resulting_states[0]

      #######################################
      for pt_number in range(resulting_states.shape[0]):

        #array of "the point"... for each sim
        pt = resulting_states[pt_number] # N x state

        #arrays of line segment points... for each sim
        curr_start = desired_states[curr_seg]
        curr_end = desired_states[curr_seg+1]
        next_start = desired_states[curr_seg+1]
        next_end = desired_states[curr_seg+2]

        #vars... for each sim
        min_perp_dist = np.ones((N, ))*5000

        ############ closest distance from point to current line segment

        #vars
        a = pt[:,x_index]- curr_start[:,0]
        b = pt[:,y_index]- curr_start[:,1]
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
        min_perp_dist = np.sqrt((pt[:,x_index]-closest_pt_x)*(pt[:,x_index]-closest_pt_x) + (pt[:,y_index]-closest_pt_y)*(pt[:,y_index]-closest_pt_y))

        #"forward-ness" of the pt... for each sim
        curr_forward = which_line_section

        ############ closest distance from point to next line segment

        #vars
        a = pt[:,x_index]- next_start[:,0]
        b = pt[:,y_index]- next_start[:,1]
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
        dist = np.sqrt((pt[:,x_index]-closest_pt_x)*(pt[:,x_index]-closest_pt_x) + (pt[:,y_index]-closest_pt_y)*(pt[:,y_index]-closest_pt_y))

        #pick which line segment it's closest to, and update vars accordingly
        curr_seg[dist<=min_perp_dist] += 1
        moved_to_next[dist<=min_perp_dist] = 1
        curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]#### np.clip(which_line_section,0,1)[dist<=min_perp_dist]
        min_perp_dist = np.min([min_perp_dist, dist], axis=0)

        '''print "\n$$$$$$$$$$$$$$$$$$$"
        print "min_perp_dist"
        print min_perp_dist[:50]
        print min_perp_dist.shape
        print "$$$$$$$$$$$$$$$$$$$\n"'''

        ################## scoring
        #penalize horiz dist
        scores += min_perp_dist*horiz_penalty_factor

        #penalize moving backward
        scores[moved_to_next==0] += (prev_forward - curr_forward)[moved_to_next==0]*backward_discouragement

        #penalize heading away from angle of line
        desired_yaw = np.arctan2(curr_end[:,1]-curr_start[:,1], curr_end[:,0]-curr_start[:,0])
        curr_yaw = np.arctan2(pt[:,yaw_sin_index],pt[:,yaw_cos_index])
        diff = np.abs(moving_distance(desired_yaw, curr_yaw))
        '''print "\n$$$$$$$$$$$$$$$$$$$"
        print "diff"
        print diff[:50]
        print diff.shape
        print "$$$$$$$$$$$$$$$$$$$\n"'''
        # diff = np.multiply(diff, (1-np.exp(-diff*2)))
        '''for i in range(diff.shape[0]):
          if min_perp_dist[i] < 0.1: diff[i] = 0
          elif min_perp_dist[i] > 0.2: diff[i] = diff[i] * 1.25'''
        scores += diff*heading_penalty_factor

        #update
        prev_forward = np.copy(curr_forward)
        prev_pt = np.copy(pt)

      #pick lowest score and the corresponding sequence of actions (out of those N)
      best_score = np.min(scores)
      best_sim_number = np.argmin(scores) 
      best_sequence = all_samples[best_sim_number]

      if(self.visualize_rviz):
        #publish the desired traj
        markerArray2 = MarkerArray()
        marker_id=0
        for des_pt_num in range(5): #5 for all, 8 for circle, 4 for zigzag
          marker = Marker()
          marker.id=marker_id
          marker.header.frame_id = "/world"
          marker.type = marker.SPHERE
          marker.action = marker.ADD
          marker.scale.x = 0.15
          marker.scale.y = 0.15
          marker.scale.z = 0.15
          marker.color.a = 1.0
          marker.color.r = 0.0
          marker.color.g = 0.0
          marker.color.b = 1.0
          marker.pose.position.x = desired_states[des_pt_num,0]
          marker.pose.position.y = desired_states[des_pt_num,1]
          marker.pose.position.z = 0
          markerArray2.markers.append(marker)
          marker_id+=1
        publish_markers_desired.publish(markerArray2)

        #publish the best sequence selected
        best_sequence_of_states= resulting_states[:,best_sim_number,:] # (h+1)x(stateSize)
        markerArray = MarkerArray()
        marker_id=0

        #print("rviz red dot state shape: ", best_sequence_of_states[0, :].shape)
        for marker_num in range(resulting_states.shape[0]):
          marker = Marker()
          marker.id=marker_id
          marker.header.frame_id = "/world"
          marker.type = marker.SPHERE
          marker.action = marker.ADD
          marker.scale.x = 0.05
          marker.scale.y = 0.05
          marker.scale.z = 0.05
          marker.color.a = 1.0
          marker.color.r = 1.0
          marker.color.g = 0.0
          marker.color.b = 0.0
          marker.pose.position.x = best_sequence_of_states[marker_num,0]
          marker.pose.position.y = best_sequence_of_states[marker_num,1]
          #print("rviz detects current roach pose to be: ", best_sequence_of_states[marker_num, :2])
          marker.pose.position.z = 0
          markerArray.markers.append(marker)
          marker_id+=1
        publish_markers.publish(markerArray)

      #the 0th entry is the currently executing action... so the 1st entry is the optimal action to take
      action_to_take = np.copy(best_sequence[dt_steps])

      #return
      return action_to_take, curr_line_segment, old_curr_forward, save_perp_dist, save_forward_dist, saved_old_forward_dist, save_moved_to_next, save_desired_heading, save_curr_heading

#distance needed for unit 2 to go toward unit1
#NOT THE CORRECT SIGN
def moving_distance(unit1, unit2):
  phi = (unit2-unit1) % (2*np.pi)

  phi[phi > np.pi] = (2*np.pi-phi)[phi > np.pi]

  return phi