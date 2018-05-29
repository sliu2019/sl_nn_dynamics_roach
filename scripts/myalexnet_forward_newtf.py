################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image, and change to BGR

# images = []
# for i in range(10):
#   im1 = (imread("/home/anagabandi/roach_workspace/src/nn_dynamics_roach/images/styrofoam_images_" + str(i) + ".jpg")[:,:,:3]).astype(float32)
#   im1 = im1 - mean(im1)
#   im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#   images.append(im1)

# for i in range(10):
#   im1 = (imread("/home/anagabandi/roach_workspace/src/nn_dynamics_roach/images/gravel_images_" + str(i) + ".jpg")[:,:,:3]).astype(float32)
#   im1 = im1 - mean(im1)
#   im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#   images.append(im1)

# for i in range(10):
#   im1 = (imread("/home/anagabandi/roach_workspace/src/nn_dynamics_roach/images/carpet_images_" + str(i) + ".jpg")[:,:,:3]).astype(float32)
#   im1 = im1 - mean(im1)
#   im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#   images.append(im1)

# for i in range(10):
#   im1 = (imread("/home/anagabandi/roach_workspace/src/nn_dynamics_roach/images/turf_images_" + str(i) + ".jpg")[:,:,:3]).astype(float32)
#   im1 = im1 - mean(im1)
#   im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#   images.append(im1)
images = []
directories = ["/home/anagabandi/Pictures/carpet_images/final", "/home/anagabandi/Pictures/styrofoam_images/final", "/home/anagabandi/Pictures/turf_images/final"]
for directory in directories:
  for filename in os.listdir(directory):
    # Preprocess by subtracting the mean, and converting from RGB to BGR
    im = (imread(directory + "/" + filename)[:,:,:3]).astype(float32)
    im = im - mean(im)
    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
    images.append(im)
#

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


print(xdim)
x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

t = time.time()
#output = sess.run(prob, feed_dict = {x:images})

def getPointsAndMean(layerOut, inDim, outDim):

  points = []
  np.random.seed(0)
  matrix = np.random.uniform(size=(inDim, outDim))
  for i in layerOut:
    points.append(i.dot(matrix))
  mean = np.zeros((4, outDim))
  for i in range(3):
    for j in range(10):
      mean[i] += points[i*10 + j]
    mean[i] *= 1.0
    mean[i] /= 10.0

  return points, mean

#output = sess.run(maxpool2, feed_dict = {x:images})
#outputTwo = sess.run(maxpool5, feed_dict = {x:images})
print("before calling convnet")
output = sess.run(fc8, feed_dict = {x: images})

out = []
for i in output:
  out.append(np.ndarray.flatten(i))
out = np.array(out)

#out2 = []
#for i in outputTwo:
#  out2.append(np.ndarray.flatten(i))

#print(out2.shape)
#out2 = np.array(out2)

points, mean = getPointsAndMean(out, len(out[0]), 10)
#points_5, mean_5 = getPointsAndMean(out2, 9216, 50)
#points_3, mean_3 = getPointsAndMean(out3, 43264 / 4, 50)
accuracy = np.zeros(4)
for k in range(3):
  for i in range(k*10, (k+1)*10):
    maxP = -1
    maxVal = 99999999
    #pointOne = points_2[i]
    #pointTwo = points_5[i]
    point = points[i]
    for j in range(4):

      #cur = np.linalg.norm(pointOne - mean_2[j])
      #cur2 = np.linalg.norm(pointTwo - mean_5[j])
      cur = np.linalg.norm(point - mean[j])
      #print(cur)
      if cur < maxVal:
        maxVal = cur
        maxP = j
    if maxP == int(i / 10):
      accuracy[k] += 1

print (accuracy)

model = TSNE(n_components=2, random_state=0)
look = model.fit_transform(points)

plt.scatter(look[:10:,0], look[:10:,1], c='r')
plt.scatter(look[10:20:,0], look[10:20:,1],c='b')
plt.scatter(look[20:30:,0], look[20:30:,1], c='g')
plt.scatter(look[30:40:,0], look[30:40:,1], c='k')
plt.show()
################################################################################

#Output:


for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind,:]
    print("Image", input_im_ind)
    for i in range(5):
        print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])

print(time.time()-t)
