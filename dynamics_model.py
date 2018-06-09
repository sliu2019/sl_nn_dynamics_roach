
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math



class Dyn_Model:

    def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize, which_agent, x_index, y_index, 
                num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, np_datatype,
                print_minimal, feedforward_network, use_one_hot, use_camera, curr_env_onehot, N, use_multistep_loss=False, one_hot_dims=4):

        

        #init vars
        self.sess = sess
        self.batchsize = batchsize
        self.which_agent = which_agent
        self.x_index = x_index
        self.y_index = y_index
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.print_minimal = print_minimal
        self.use_multistep_loss = use_multistep_loss
        self.np_datatype = np_datatype
        self.use_one_hot = use_one_hot
        self.use_camera = use_camera
        self.one_hot_dims=one_hot_dims

        self.curr_env_onehot = curr_env_onehot
        #set curr_env_onehot
        # if(self.use_one_hot):
        #     self.curr_env_onehot = np.tile(curr_env_onehot, (N,1))
        # else:
        #     self.curr_env_onehot= np.ones((1,self.one_hot_dims))

        #placeholders
        self.x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') #inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #labels
        self.next_z_ = tf.placeholder(tf_datatype, shape=[None, 3, self.outputSize], name='next_z')
        #forward pass
        if self.use_one_hot and self.use_camera:
            self.tiled_camera_input = tf.placeholder(tf_datatype, shape=[None, 227, 227, 3])
            self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize, 
                                                    num_fc_layers, depth_fc_layers, tf_datatype, self.tiled_camera_input)
        else:
            self.tiled_onehots = tf.placeholder(tf_datatype, shape=[None, self.one_hot_dims]) #tiled one hot vectors
            self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize, 
                                                    num_fc_layers, depth_fc_layers, tf_datatype, self.tiled_onehots)

        if self.use_multistep_loss:
            self.nn_output2 = feedforward_network(self.curr_nn_output, self.inputSize, self.outputSize, 
                                                        num_fc_layers, depth_fc_layers, tf_datatype)
            self.nn_output3 = feedforward_network(self.nn_output2, self.inputSize, self.outputSize, 
                                                        num_fc_layers, depth_fc_layers, tf_datatype)
            self.nn_output4 = feedforward_network(self.nn_output3, self.inputSize, self.outputSize, 
                                                        num_fc_layers, depth_fc_layers, tf_datatype)
            # loss
            self.mse_ = tf.reduce_mean(tf.add_n([tf.square(self.z_ - self.curr_nn_output), tf.square(self.next_z_[:,0] - self.nn_output2),
                                                tf.square(self.next_z_[:,1] - self.nn_output3), tf.square(self.next_z_[:,2] - self.nn_output4)]))
        else:
            self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))

        # Compute gradients and update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.theta = tf.trainable_variables()
        self.gv = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

    def train(self, dataX, dataZ, dataOneHots, dataCamera, dataX_new, dataZ_new, nEpoch, save_dir, fraction_use_new):

        '''TO DO: new data doesnt have corresponding onehots right now'''

        #init vars
        start = time.time()
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData_old = dataX.shape[0]
        num_new_pts = dataX_new.shape[0]

        #how much of new data to use per batch
        if(num_new_pts<(self.batchsize*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones
        else:
            batchsize_new_pts = int(self.batchsize*fraction_use_new)

        #how much of old data to use per batch
        batchsize_old_pts = int(self.batchsize- batchsize_new_pts)

        #training loop
        for i in range(nEpoch):
            
            #reset to 0
            avg_loss=0
            num_batches=0

            #randomly order indeces (equivalent to shuffling dataX and dataZ)
            old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
            #train from both old and new dataset
            if(batchsize_old_pts>0): 

                #get through the full old dataset
                for batch in range(int(math.floor(nData_old / batchsize_old_pts))):

                    #randomly sample points from new dataset
                    if(num_new_pts==0):
                        dataX_new_batch = dataX_new
                        dataZ_new_batch = dataZ_new
                    else:
                        new_indeces = npr.randint(0,dataX_new.shape[0], (batchsize_new_pts,))
                        dataX_new_batch = dataX_new[new_indeces, :]
                        dataZ_new_batch = dataZ_new[new_indeces, :]

                    #walk through the randomly reordered "old data"
                    dataX_old_batch = dataX[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    dataZ_old_batch = dataZ[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    
                    #combine the old and new data
                    dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
                    dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))

                    if self.use_one_hot and self.use_camera: # Live camera
                        #Hasn't been tiled yet due to memory constraints
                        steps_per_rollout = dataX.shape[0]/dataCamera.shape[0]
                        print(steps_per_rollout)
                        print(type(old_indeces))
                        actual_indices = np.floor(old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts]/steps_per_rollout)
                        dataCamera_batch = dataCamera[actual_indices, :, :, :] 
                    else:
                        dataOneHots_batch = dataOneHots[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    

                    data_next_indeces = np.clip(np.array(old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts]) + 1, 0, dataX.shape[0]-1)
                    data_next_indeces = np.clip([data_next_indeces + 1, data_next_indeces + 2, data_next_indeces + 3], 0, dataX.shape[0]-1).T
                    dataZ_next = dataZ[data_next_indeces, :]

                    #one iteration of feedforward training
                    if self.use_one_hot and self.use_camera:
                        _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch,
                                                                self.next_z_: dataZ_next, self.tiled_camera_input: dataCamera_batch})
                    else:
                        _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch,
                                                                self.next_z_: dataZ_next, self.tiled_onehots: dataOneHots_batch})

                    training_loss_list.append(loss)
                    avg_loss+= loss
                    num_batches+=1

            #save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if(not(self.print_minimal)):
                if((i%10)==0):
                    print("\n=== Epoch {} ===".format(i))
                    print ("loss: ", avg_loss/num_batches)
        
        if(not(self.print_minimal)):
            print ("Training set size: ", (nData_old + dataX_new.shape[0]))
            print("Training duration: {:0.2f} s".format(time.time()-start))

        #get loss of curr model on old dataset
        avg_old_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(nData_old / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ[batch*self.batchsize:(batch+1)*self.batchsize, :]
            if dataOneHots:
                dataOneHots_batch = dataOneHots[batch*self.batchsize:(batch+1)*self.batchsize, :]
            elif dataCamera:
                dataCamera_batch = dataCamera[batch*self.batchsize:(batch+1)*self.batchsize, :]

            data_next_indeces = np.clip(np.array(old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts]) + 1, 0, dataX.shape[0]-1)
            data_next_indeces = np.clip([data_next_indeces + 1, data_next_indeces + 2, data_next_indeces + 3], 0, dataX.shape[0]-1).T
            dataZ_next = dataZ[data_next_indeces, :]
            #one iteration of feedforward training

            if dataOneHots:
                loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch, self.next_z_: dataZ_next, 
                                                                                self.tiled_onehots: dataOneHots_batch})
            elif dataCamera:
                loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch, self.next_z_: dataZ_next, 
                                                                                self.tiled_camera_input: dataCamera_batch})

            avg_old_loss+= loss
            iters_in_batch+=1
        old_loss =  avg_old_loss/iters_in_batch

        #get loss of curr model on new dataset
        avg_new_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(dataX_new.shape[0] / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ_new[batch*self.batchsize:(batch+1)*self.batchsize, :]

            if dataOneHots:
                dataOneHots_batch = dataOneHots[batch*self.batchsize:(batch+1)*self.batchsize, :]
            elif dataCamera:
                dataCamera_batch = dataCamera[batch*self.batchsize:(batch+1)*self.batchsize, :]

            data_next_indeces = np.clip(np.array(old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts]) + 1, 0, dataX.shape[0]-1)
            data_next_indeces = np.clip([data_next_indeces + 1, data_next_indeces + 2, data_next_indeces + 3], 0, dataX.shape[0]-1).T
            dataZ_next = dataZ[data_next_indeces, :]
            #one iteration of feedforward training
            if dataOneHots:
                loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch, self.next_z_: dataZ_next, 
                                                                                self.tiled_onehots: dataOneHots_batch})
            elif dataCamera:
                loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch, self.next_z_: dataZ_next, 
                                                                                self.tiled_camera_input: dataCamera_batch})
            avg_new_loss+= loss
            iters_in_batch+=1
        if(iters_in_batch==0):
            new_loss=0
        else:
            new_loss =  avg_new_loss/iters_in_batch

        #done
        return (avg_loss/num_batches), old_loss, new_loss

    ##### TO DO: implement onehot stuff here
    '''def run_validation(self, inputs, outputs):

        #init vars
        nData = inputs.shape[0]
        avg_loss=0
        iters_in_batch=0

        for batch in range(int(math.floor(nData / self.batchsize))):
            # Batch the training data
            dataX_batch = inputs[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = outputs[batch*self.batchsize:(batch+1)*self.batchsize, :]

            #one iteration of feedforward training
            z_predictions, loss = self.sess.run([self.curr_nn_output, self.mse_], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch, self.tiled_onehots: tiled_onehots})

            avg_loss+= loss
            iters_in_batch+=1

        #avg loss + all predictions
        print ("Validation set size: ", nData)
        print ("Validation set's total loss: ", avg_loss/iters_in_batch)

        return (avg_loss/iters_in_batch)'''

    #multistep prediction using the learned dynamics model at each step
    def do_forward_sim(self, forwardsim_x_true, forwardsim_y, forwardsim_onehot, img, many_in_parallel, env_inp, which_agent):
        # forwardsim_x_true: actual state sequence: should be a full state representation
        # forwardsim_y : action sequence
        # forwardsim_onehot : only used for the 1 forward sim in train_dynamics
        # img: camera image : 
        # many_in_parallel: evaluate many action sequences simultaneously just using matrix-vector notation
        #init vars
        state_list = []

        if(many_in_parallel):
            #init vars
            N= forwardsim_y.shape[0]
            
            if(self.use_one_hot):
                if self.use_camera:
                    # Q: Is it also just np.tile for 3D matrix inputs to a CNN?
                    self.tiled_img = np.tile(img, (N, 1))
                else:
                    self.tiled_curr_env_onehot = np.tile(self.curr_env_onehot, (N,1))
            else:
                self.tiled_curr_env_onehot= np.ones((1,self.one_hot_dims))
            
            horizon = forwardsim_y.shape[1]
            array_stdz = np.tile(np.expand_dims(self.std_z, axis=0),(N,1))
            array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0),(N,1))
            array_stdy = np.tile(np.expand_dims(self.std_y, axis=0),(N,1))
            array_meany = np.tile(np.expand_dims(self.mean_y, axis=0),(N,1))
            array_stdx = np.tile(np.expand_dims(self.std_x, axis=0),(N,1))
            array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0),(N,1))

            if(len(forwardsim_x_true)==2):
                #N starting states, one for each of the simultaneous sims
                curr_states=np.tile(forwardsim_x_true[0], (N,1))
            else:
                curr_states=np.copy(forwardsim_x_true)

            #advance all N sims, one timestep at a time
            for timestep in range(horizon):

                #keep track of states for all N sims
                state_list.append(np.copy(curr_states))

                #make [N x (state,action)] array to pass into NN
                # Assuming that the position (x, y, z) are in the 1st three entries of state
                #print("inside forward_sim, the curr_state shape is: ", curr_states.shape)
                abbrev_curr_states = curr_states[:, 3:]
                states_preprocessed = np.nan_to_num(np.divide((abbrev_curr_states-array_meanx), array_stdx))
                actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
                inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

                #run the N sims all at once
                if self.use_one_hot and self.use_camera:
                    model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list, self.tiled_camera_input: self.tiled_img}) 
                else:
                    model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list, self.tiled_onehots: self.tiled_curr_env_onehot}) 
                state_differences = np.multiply(model_output[0],array_stdz)+array_meanz

                #update the state info
                curr_states = curr_states + state_differences

            #return a list of length = horizon+1... each one has N entries, where each entry is (13,)
            state_list.append(np.copy(curr_states))
        else:

            curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input
            curr_index = 0

            for curr_control in forwardsim_y:

                state_list.append(np.copy(curr_state))
                curr_control = np.expand_dims(curr_control, axis=0)

                if(self.use_one_hot):
                    if self.use_camera:
                        # nothing to do here, currently
                        img = img
                    else:
                        curr_onehot = np.expand_dims(forwardsim_onehot[curr_index], axis=0) 
                else:
                    curr_onehot= np.ones((1,self.one_hot_dims))

                #subtract mean and divide by standard deviation
                #print("inside forward sim, and the dimension of curr_state is: ", curr_state.shape)
                abbrev_curr_state = curr_state[3:]
                curr_state_preprocessed = abbrev_curr_state - self.mean_x
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
                curr_control_preprocessed = curr_control - self.mean_y
                curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/self.std_y)
                inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

                #run through NN to get prediction
                if self.use_one_hot and self.use_camera:
                    model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed, self.tiled_camera_input: img}) 
                else:
                    model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed, self.tiled_onehots: curr_onehot}) 

                #multiply by std and add mean back in
                state_differences= (model_output[0][0]*self.std_z)+self.mean_z

                #update the state info
                next_state = curr_state + state_differences

                #copy the state info
                curr_state= np.copy(next_state)
                curr_index+=1

            state_list.append(np.copy(curr_state))
              
        return state_list