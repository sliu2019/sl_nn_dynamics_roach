
from velociroach import *
import shared_multi as shared
from Queue import Queue
from collections import OrderedDict
import math
import copy
import tensorflow as tf

def setup_roach(serial_port, baud_rate, DEFAULT_ADDRS, use_pid_mode, top):

	#setup serial
	xb = None
	try:
		xb = setupSerial(serial_port, baud_rate)
		print("Done setting up serial.\n")
	except:
		print('Failed to set up serial, exiting')

	#setup the roach
	if xb is not None:
		shared.xb = xb
		robots = [Velociroach(addr,xb) for addr in DEFAULT_ADDRS]
		n_robots=len(robots)

		for r in robots:
			r.running = False
			r.VERBOSE = False
			r.setPIDOutputChannel(1)
			if(use_pid_mode):
				r.PIDStartMotors()
				r.running = True
			r.zeroPosition() ############### ????


			# r.setupTelemetryDataNum(5000)
			# time.sleep(1)
			# r.eraseFlashMem()
			# time.sleep(1)
			# r.startTelemetrySave()
		shared.ROBOTS = robots

		#setup the info receiving
		shared.imu_queues = OrderedDict()
		shared.imu_queues[robots[0].DEST_ADDR_int] = Queue()

		print("Done setting up RoachBridge.\n")
	print(xb)
	return xb, robots, shared.imu_queues

def start_fans(lock, robot):
	lock.acquire()
	for i in range(3):
		robot.startFans()
		time.sleep(1)
	lock.release()

def stop_fans(lock, robot):
	lock.acquire()
	for i in range(3):
		robot.stopFans()
		time.sleep(1)
	lock.release()


def start_roach(xb, lock, robots, use_pid_mode):

	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			robot.PIDStartMotors()
			robot.running = True
		robot.setThrustGetTelem(0, 0) 

		
	lock.release()
	return

def stop_roach(lock, robots, use_pid_mode):

	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			robot.setVelGetTelem(0,0)
			#robot.PIDStopMotors()
			#robot.running = False
		else:
			robot.setThrustGetTelem(0, 0)
			#robot.downloadTelemetry() 
	lock.release()
	return

def stop_and_exit_roach(xb, lock, robots, use_pid_mode):

	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			#robot.setVelGetTelem(0,0)
			robot.PIDStopMotors()
			robot.running = False
		else:
			robot.setThrustGetTelem(0, 0) 
			### robot.downloadTelemetry()
	lock.release()

	#exit RoachBridge
	xb_safe_exit(xb)
	return

def quat_to_eulerDegrees(orientation):
  x=orientation.x
  y=orientation.y
  z=orientation.z
  w=orientation.w

  ysqr = y*y
  
  t0 = +2.0 * (w * x + y*z)
  t1 = +1.0 - 2.0 * (x*x + ysqr)
  X = math.degrees(math.atan2(t0, t1))
  
  t2 = +2.0 * (w*y - z*x)
  t2 =  1 if t2 > 1 else t2
  t2 = -1 if t2 < -1 else t2
  Y = math.degrees(math.asin(t2))
  
  t3 = +2.0 * (w * z + x*y)
  t4 = +1.0 - 2.0 * (ysqr + z*z)
  Z = math.degrees(math.atan2(t3, t4))
  
  return [X,Y,Z] 

#datatypes
tf_datatype= tf.float32
np_datatype= np.float32

mappings = np.load("images.npy")

def create_onehot(curr_surface, use_camera = False, mappings= None):
    curr_onehot = None

    if (use_camera):

        index = 0
        if(curr_surface=='carpet'):
            index = 0
        if(curr_surface=='gravel'):
            index = 20
        if(curr_surface=='turf'):
            index = 30
        if(curr_surface=='styrofoam'):
            index = 10
        index += np.random.randint(10)
        curr_onehot = mappings[index]
        curr_onehot=np.array(list(curr_onehot) + [1])


        #mean vec (used to do this with subtracting old mean, rather than new mean in myalexnet)
        '''mean_carpet = np.mean(mappings[0:10,:], axis=0)
        mean_gravel = np.mean(mappings[10:20,:], axis=0)
        mean_turf = np.mean(mappings[20:30,:], axis=0)
        mean_sty = np.mean(mappings[30:40,:], axis=0)

        if(curr_surface=='carpet'):
            curr_onehot = mean_carpet
        if(curr_surface=='gravel'):
            curr_onehot = mean_gravel
        if(curr_surface=='turf'):
            curr_onehot = mean_turf
        if(curr_surface=='styrofoam'):
            curr_onehot = mean_sty
        curr_onehot=np.array(list(curr_onehot) + [1])'''
        
    else:
        #use one hot
        curr_onehot = np.zeros((1,4)).astype(np_datatype)
        if(curr_surface=='carpet'):
            curr_onehot[0,0]=1
        if(curr_surface=='gravel'):
            curr_onehot[0,1]=1
        if(curr_surface=='turf'):
            curr_onehot[0,2]=1
        if(curr_surface=='styrofoam'):
            curr_onehot[0,3]=1

    return curr_onehot

def singlestep_to_state(robot_info, mocap_info, old_time, old_pos, old_al, old_ar, state_representation):

	#dt
	curr_time = robot_info.stamp
	if(old_time==-7):
		dt=1
	else:
		dt = (curr_time.secs-old_time.secs) + (curr_time.nsecs-old_time.nsecs)*0.000000001

	#mocap position
	curr_pos= mocap_info.pose.position

	#mocap pose
	angles = quat_to_eulerDegrees(mocap_info.pose.orientation)
	r= angles[0]
	p= angles[1]
	yw= angles[2]
	#convert r,p,y to rad
	r=r*np.pi/180.0
	p=p*np.pi/180.0
	yw=yw*np.pi/180.0

	#gyro angular velocity
	wx= robot_info.gyroX
	wy= robot_info.gyroY
	wz= robot_info.gyroZ

	#encoders
	al= robot_info.posL/math.pow(2,16)*2*math.pi
	ar= robot_info.posR/math.pow(2,16)*2*math.pi

	#com vel
	vel_x = (curr_pos.x-old_pos.x)/dt
	vel_y = (curr_pos.y-old_pos.y)/dt
	vel_z = (curr_pos.z-old_pos.z)/dt

	#motor vel
	vel_al = (al-old_al)/dt
	vel_ar = (ar-old_ar)/dt

	#create the state
	if(state_representation=="all"):
		state = np.array([curr_pos.x, curr_pos.y, curr_pos.z, 
									vel_x, vel_y, vel_z, 
									np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
									wx, wy, wz, 
									np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
									vel_al, vel_ar, 
									robot_info.bemfL, robot_info.bemfR, robot_info.vBat])

	#save curr as old
	old_time= copy.deepcopy(curr_time)
	old_pos=copy.deepcopy(curr_pos)
	old_al=np.copy(al)
	old_ar=np.copy(ar)

	return state, old_time, old_pos, old_al, old_ar

def rollout_to_states(robot_info, mocap_info, state_representation):

	list_states=[]
	list_actions=[]

	for step in range(0,len(robot_info)):

		if(step==0):
			old_time= robot_info[step].stamp
			old_pos= mocap_info[step].pose.position
			old_al= robot_info[step].posL/math.pow(2,16)*2*math.pi
			old_ar= robot_info[step].posR/math.pow(2,16)*2*math.pi
		else:
			#dt
			curr_time = robot_info[step].stamp
			dt = (curr_time.secs-old_time.secs) + (curr_time.nsecs-old_time.nsecs)*0.000000001

			#mocap position
			curr_pos= mocap_info[step].pose.position

			#mocap pose
			angles = quat_to_eulerDegrees(mocap_info[step].pose.orientation)
			r= angles[0]
			p= angles[1]
			yw= angles[2]
			#convert r,p,y to rad
			r=r*np.pi/180.0
			p=p*np.pi/180.0
			yw=yw*np.pi/180.0

			#gyro angular velocity
			wx= robot_info[step].gyroX
			wy= robot_info[step].gyroY
			wz= robot_info[step].gyroZ

			#encoders
			al= robot_info[step].posL/math.pow(2,16)*2*math.pi
			ar= robot_info[step].posR/math.pow(2,16)*2*math.pi

			#com vel
			vel_x = (curr_pos.x-old_pos.x)/dt
			vel_y = (curr_pos.y-old_pos.y)/dt
			vel_z = (curr_pos.z-old_pos.z)/dt

			#motor vel
			vel_al = (al-old_al)/dt
			vel_ar = (ar-old_ar)/dt

			#create the state
			if(state_representation=="all"):
				states = np.array([curr_pos.x, curr_pos.y, curr_pos.z, 
											vel_x, vel_y, vel_z, 
											np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
											wx, wy, wz, 
											np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
											vel_al, vel_ar, 
											robot_info[step].bemfL, robot_info[step].bemfR, robot_info[step].vBat])
				list_states.append(states)

			#create the action
			action=np.array([robot_info[step].curLeft, robot_info[step].curRight])
			list_actions.append(action)

			#save curr as old
			old_time=copy.deepcopy(curr_time)
			old_pos=copy.deepcopy(curr_pos)
			old_al=copy.deepcopy(al)
			old_ar=copy.deepcopy(ar)
	return np.array(list_states), np.array(list_actions)