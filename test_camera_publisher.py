
#!usr/bin/env python
import rospy 
from rospy.msg import numpy_msg
from std_msgs.msg import Int32, Float32, String, Float32MultiArray

try:
	msg = rospy.wait_for_message("live_camera_image", numpy_msg(Float32MultiArray), timeout=0.05)
	print(type(msg))
except ROSException:
	# Use previous image 
	pass 
