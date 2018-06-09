#!/usr/bin/env python

# If you want to do it simply, poll camera until an image is returned (bool for success is true)
# Otherwise, you will need to poll the roach/mocap too, so figure out way to poll both at once
# It seems grabbing the image never fails, unless camera is off or feed has stopped

# Oh fuck: you want to get image and then preprocess it in the way you preprocessed inputs for k-means: you want to have .jpg and a 227x227 image
# Maybe you can change the camera feed to be 227x227? Is that a property you can change? And if you did, wouldn't that make a rectangular image a square image in an arbitrary manner?

import numpy as np
import cv2
from os import system
from scipy.misc import imread

for i in range(3):
	cap = cv2.VideoCapture(1)
	ret, frame = cap.read()

	cv2.imshow('frame', frame)
	# A temporary file that it's ok to continuously write over
	# "Image" object in OpenCV to .jpg
	temp_img_filename = "frame.jpg"
	cv2.imwrite(temp_img_filename, frame)
	# Crop to correct size
	#system('convert ' + filepath + ' ' + filepath[:-4] + '.jpg')
	system('convert ' + temp_img_filename + ' -crop 480x480+80+0 ' + temp_img_filename + '_cropped.jpg')
	system('convert ' + temp_img_filename + '_cropped.jpg' + ' -resize 227x227 ' + temp_img_filename + '_final.jpg')

	# **********PREPROCESS**************
	# Subtract mean, flip rgb to bgr, then feed into alexnet + random projection + feedforwardnetwork_camera 
	# This should be the mean of the dataset alexnet was trained on....
	training_mean= [123.68, 116.779, 103.939]

	im = (imread(temp_img_filename)[:,:,:3]).astype(float32)

	im = im - training_mean 
	im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
	#images.append(im)

	# **************EVALUATE ON ALEXNET AND NEURAL NET**************
	# You can test the NN here by using your tune_hyperparam.py code

	# Wait so when we do k-random offshoots, we use the current image throughout the action sequence rollout?

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cap.release()



# 640 X 480