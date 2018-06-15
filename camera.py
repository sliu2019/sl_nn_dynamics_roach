#!usr/bin/env python

#remove or add the library/libraries for ROS
import rospy, time, math, random
from rospy.msg import numpy_msg
#remove or add the message type
from std_msgs.msg import Int32, Float32, String, Float32MultiArray
from basics.msg import TimerAction, TimerGoal, TimeResult
from time import sleep
import cv2
from scipy.misc import imresize

def talker(camera_serial_port):
	camera_serial_port = camera_serial_port
	pub = rospy.Publisher("live_camera_image", numpy_msg(Float32MultiArray), queue_size=1)
	rospy.init_node("camera")

	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		cap = cv2.VideoCapture(int(camera_serial_port[-1]))
		ret, frame = cap.read()
		
		frame_cropped = frame[:, 80:560, :]
		img = imresize(frame_cropped, (227, 227, 3))
		img = np.swapaxes(img, 0, 1)

		training_mean= [123.68, 116.779, 103.939] 
		img = img - training_mean 
		img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]

		cap.release()

		pub.publish(img)
		rospy.loginfo()
		rate.sleep()


if __name__=='__main__':
	camera_serial_port = "/dev/video1"
	try:
		talker(camera_serial_port)
	except rospy.ROSInterruptException:
		pass




"""import rospy
   4 from std_msgs.msg import String
   5 
   6 def talker():
   7     pub = rospy.Publisher('chatter', String, queue_size=10)
   8     rospy.init_node('talker', anonymous=True)
   9     rate = rospy.Rate(10) # 10hz
  10     while not rospy.is_shutdown():
  11         hello_str = "hello world %s" % rospy.get_time()
  12         rospy.loginfo(hello_str)
  13         pub.publish(hello_str)
  14         rate.sleep()
  15 
  16 if __name__ == '__main__':
  17     try:
  18         talker()
  19     except rospy.ROSInterruptException:
  20         pass"""


"""Listener example:

   from rospy.numpy_msg import numpy_msg

   rospy.init_node('mynode')
   rospy.Subscriber("mytopic", numpy_msg(TopicType)
Publisher example:

   from rospy.numpy_msg import numpy_msg
   import numpy
   
   pub = rospy.Publisher('mytopic', numpy_msg(TopicType))
   rospy.init_node('mynode')
   a = numpy.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=numpy.float32)
   pub.publish(a)"""