
import numpy as np
import tensorflow as tf

'''
OLD NETWORK: 
    inp x 500 --> 1x500
    500 x 500 --> 1x500
    500 x out --> 1xout

NEW NETWORK:
    inp x 500 --> 1x500
    (501*5) x 500 --> 1x500
    500 x out --> 1xout

    fused:
        ([u 1]' [v 1]).ravel
        (501*5)
'''

def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype, tiled_onehots):

    #vars
    intermediate_size= 250 #########depth_fc_layers
    reuse= False
    initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf_datatype)
    flatten = tf.contrib.layers.flatten
    fc = tf.contrib.layers.fully_connected

    #1st hidden layer
    fc_1 = fc(inputState, num_outputs=intermediate_size, activation_fn=None, 
            weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
    h_1 = tf.nn.relu(fc_1)

    #fuse
        #h_1 is [bs x 500]
        #tiled_onehots is [bs x 4]
    u= tf.transpose(h_1)
    v= tiled_onehots   
    u_batch = tf.expand_dims(u, 2) #[500 x bs x 1]
    u_batch = tf.transpose(u_batch, [1, 0, 2]) #[bs x 500 x 1]
    v_batch = tf.expand_dims(v, 1) #[bs x 1 x 4]  
    fuse = flatten(tf.matmul(u_batch, v_batch)) #[bs x 500 x 4]  --> [bs x 2000]  

    #2nd hidden layer
    fc_2 = fc(fuse, num_outputs=intermediate_size, activation_fn=None, 
            weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
    h_2 = tf.nn.relu(fc_2)

    # make output layer
    z=fc(h_2, num_outputs=outputSize, activation_fn=None, weights_initializer=initializer, 
        biases_initializer=initializer, reuse=reuse, trainable=True)
    
    return z