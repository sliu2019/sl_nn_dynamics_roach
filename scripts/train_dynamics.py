
import pprint
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import IPython
import math
import matplotlib.pyplot as plt
import pickle
import threading
import multiprocessing
import os
import sys
from six.moves import cPickle
from scipy.signal import savgol_filter
from scipy.signal import medfilt


#add nn_dynamics_roach to sys.path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#my imports
from nn_dynamics_roach.msg import velroach_msg
from utils import *
from dynamics_model import Dyn_Model
from controller_class import Controller
from controller_class_playback import ControllerPlayback

'''
TO DO:

-multistep loss
-multiple inputs

-leave out certain elements of state, and see if training is better
-smooth vals (x/y/z/al/ar) before differentiating, and see if training is better 

'''


def main():

    ##################################
    ######### SPECIFY VARS ###########
    ##################################

    #saving directories
    run_num=0                                          #directory for saving everything
    desired_shape_for_traj = "straight"                     #straight, left, right, circle_left, zigzag, figure8
    traj_save_path= desired_shape_for_traj + str(0)     #directory name inside run_num directory
    task_type=['carpet']                                 # "all" if want to use all data
    
    #running
    use_pid_mode = True 
    slow_pid_mode = False
    serial_port = '/dev/ttyUSB0'
    baud_rate = 57600
    DEFAULT_ADDRS = ['\x00\x01']

    #training data
    # filename_trainingdata='/data_collection/carpet_2018_01_18_12_08_08'
    # training_rollouts=np.arange(9)+0
    # validation_rollouts=[9]
    training_ratio = 0.9
    data_path = os.path.abspath(os.path.join(os.getcwd(), "../data_collection/"))
    path_lst = []

    for subdir, dirs, files in os.walk(data_path):
        surface = subdir.split("/")[-1].split("_")[0]
        if surface in task_type or task_type == "all":
            for file in files:
                path_lst.append(os.path.join(subdir, file))
    path_lst.sort()
    training_rollouts = int(len(path_lst)*training_ratio)
    if training_rollouts%2 != 0:
        training_rollouts -= 1
    validation_rollouts = len(path_lst) - training_rollouts


    #training
    use_existing_data = True
    train_now = False
    nEpoch_initial = 75
    nEpoch = 20
    state_representation = "all" #all, only_yaw, no_legs, no_legs_gyro, only_mocap
    num_fc_layers = 2
    depth_fc_layers = 500
    batchsize = 10 ######1000
    lr = 0.001
    
    #controller
    num_steps_per_controller_run= 40 #70
    N=1000
    horizon = 4
    frequency_value=10
    playback_mode = False
    
    #aggregation
    fraction_use_new = 0.5
    num_aggregation_iters = 1
    num_trajectories_for_aggregation= 1
    rollouts_forTraining = num_trajectories_for_aggregation

    ##################################
    ######### MOTOR LIMITS ###########
    ##################################

    #set min and max
    left_min = 1500
    right_min = 1500
    left_max = 3000
    right_max = 3000

    if(use_pid_mode):
      if(slow_pid_mode):
        left_min = 2*math.pow(2,16)*0.001
        right_min = 2*math.pow(2,16)*0.001
        left_max = 6*math.pow(2,16)*0.001
        right_max = 6*math.pow(2,16)*0.001
      else:
        left_min = 6*math.pow(2,16)*0.001
        right_min = 6*math.pow(2,16)*0.001
        left_max = 12*math.pow(2,16)*0.001
        right_max = 12*math.pow(2,16)*0.001
    
    min_motor_gain=0 ##??
    max_motor_gain=0 ##??

    ##################################
    ######### LOG DIRECTORY ##########
    ##################################

    #directory from which to get training data
    # data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + filename_trainingdata

    #directories for saving data
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/run_'+ str(run_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/losses')
        os.makedirs(save_dir+'/models')
        os.makedirs(save_dir+'/data')
        os.makedirs(save_dir+'/saved_forwardsim')
        os.makedirs(save_dir+'/saved_trajfollow')
        os.makedirs(save_dir+'/'+traj_save_path)
    if not os.path.exists(save_dir+'/'+traj_save_path):
        os.makedirs(save_dir+'/'+traj_save_path)

    ##return

    ##############################
    ### init vars 
    ##############################

    visualize_True = True
    visualize_False = False
    noise_True = True
    noise_False = False
    dt_steps= 1
    tf_datatype= tf.float32
    x_index=0
    y_index=1
    z_index=2

    make_aggregated_dataset_noisy = True
    make_training_dataset_noisy = True
    perform_forwardsim_for_vis= True
    print_minimal=False

    noiseToSignal = 0
    if(make_training_dataset_noisy):
        noiseToSignal = 0.01

    # num_rollouts_val = len(validation_rollouts)
    num_rollouts_val = validation_rollouts


    #################################################
    ### save a file of param values to the run directory
    #################################################

    '''param_file = open(save_dir + '/params.txt', "w")
    param_file.write("\ntrain_separate_nns = " + str(train_separate_nns))
    param_file.close()'''

    #################################################
    ### set GPU options for TF
    #################################################

    gpu_device = 0
    gpu_frac = 0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:

        #################################################
        ### read in training dataset
        #################################################

        if(use_existing_data):
            #training data
            dataX = np.load(save_dir+ '/data/dataX.npy')
            dataY = np.load(save_dir+ '/data/dataY.npy')
            dataZ = np.load(save_dir+ '/data/dataZ.npy')

            #validation data
            states_val = np.load(save_dir+ '/data/states_val.npy')
            controls_val = np.load(save_dir+ '/data/controls_val.npy')

            #data saved for forward sim
            forwardsim_x_true = np.load(save_dir+ '/data/forwardsim_x_true.npy')
            forwardsim_y = np.load(save_dir+ '/data/forwardsim_y.npy')

        else:

            ######################################
            ############ TRAINING DATA ###########
            ######################################

            dataX=[]
            dataY=[]
            dataZ=[]
            # for rollout_counter in training_rollouts:
            for i in range(training_rollouts/2):

                #read in data from 1 rollout
                # robot_file= data_dir + "/" + str(rollout_counter) + '_robot_info.obj'
                # mocap_file= data_dir + "/" + str(rollout_counter) + '_mocap_info.obj'
                mocap_file = path_lst[2*i]
                robot_file = path_lst[2*i+1]
                robot_info = pickle.load(open(robot_file,'r'))
                mocap_info = pickle.load(open(mocap_file,'r'))

                #turn saved rollout into s
                states_for_dataX, actions_for_dataY= rollout_to_states(robot_info, mocap_info, state_representation) 
                    #states_for_dataX: (length-1)x24 cuz ignore 1st one (no deriv)
                    #actions_for_dataY: (length-1)x2

                #use s to create ds
                states_for_dataZ = states_for_dataX[1:,:]-states_for_dataX[:-1,:]

                #s,a,ds
                dataX.append(states_for_dataX[:-1,:]) #the last one doesnt have a corresponding next state
                dataY.append(actions_for_dataY[:-1,:])
                dataZ.append(states_for_dataZ)
            
            #save training data
            dataX=np.concatenate(dataX)
            dataY=np.concatenate(dataY)
            dataZ=np.concatenate(dataZ)
            np.save(save_dir+ '/data/dataX.npy', dataX)
            np.save(save_dir+'/data/dataY.npy', dataY)
            np.save(save_dir+ '/data/dataZ.npy', dataZ)

            ######################################
            ########## VALIDATION DATA ###########
            ######################################

            states_val = []
            controls_val = []
            # for rollout_counter in validation_rollouts:
            for i in range(validation_rollouts/2):

                #read in data from 1 rollout
                # robot_file= data_dir + "/" + str(rollout_counter) + '_robot_info.obj'
                # mocap_file= data_dir + "/" + str(rollout_counter) + '_mocap_info.obj'
                mocap_file = path_lst[training_rollouts + 2*i]
                robot_file = path_lst[training_rollouts + 2*i+1]
                robot_info = pickle.load(open(robot_file,'r'))
                mocap_info = pickle.load(open(mocap_file,'r'))

                #turn saved rollout into s
                states_for_dataX, actions_for_dataY= rollout_to_states(robot_info, mocap_info, state_representation)

                states_val.append(states_for_dataX)
                controls_val.append(actions_for_dataY)

            #save validation data
            states_val = np.array(states_val)
            controls_val = np.array(controls_val)
            np.save(save_dir+ '/data/states_val.npy', states_val)
            np.save(save_dir+ '/data/controls_val.npy', controls_val)

            #set aside un-preprocessed data, to use later for forward sim
            forwardsim_x_true = states_for_dataX[4:16] #20:50
            forwardsim_y = actions_for_dataY[4:16] #20:50
            np.save(save_dir+ '/data/forwardsim_x_true.npy', forwardsim_x_true)
            np.save(save_dir+ '/data/forwardsim_y.npy', forwardsim_y)

        #################################################
        ### preprocess the old training dataset
        #################################################

        print("\n#####################################")
        print("Preprocessing 'old' training data")
        print("#####################################\n")
        #every component (i.e. x position) will now be mean 0, std 1

        mean_x = np.mean(dataX, axis = 0)
        dataX = dataX - mean_x
        std_x = np.std(dataX, axis = 0)
        dataX = np.nan_to_num(dataX/std_x)

        mean_y = np.mean(dataY, axis = 0) 
        dataY = dataY - mean_y
        std_y = np.std(dataY, axis = 0)
        dataY = np.nan_to_num(dataY/std_y)

        mean_z = np.mean(dataZ, axis = 0) 
        dataZ = dataZ - mean_z
        std_z = np.std(dataZ, axis = 0)
        dataZ = np.nan_to_num(dataZ/std_z)

        #save mean and std to files for controller to use
        np.save(save_dir+ '/data/mean_x.npy', mean_x)
        np.save(save_dir+ '/data/mean_y.npy', mean_y)
        np.save(save_dir+ '/data/mean_z.npy', mean_z)
        np.save(save_dir+ '/data/std_x.npy', std_x)
        np.save(save_dir+ '/data/std_y.npy', std_y)
        np.save(save_dir+ '/data/std_z.npy', std_z)

        ## concatenate state and action, to be used for training dynamics
        inputs = np.concatenate((dataX, dataY), axis=1)
        outputs = np.copy(dataZ)

        #dimensions
        assert inputs.shape[0] == outputs.shape[0]
        numData = inputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]

        ##############################################
        ########### THE DYNAMICS MODEL ###############
        ##############################################
    
        #initialize model
        dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, 0, x_index, y_index, num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal)

        #randomly initialize all vars
        sess.run(tf.initialize_all_variables())  ##sess.run(tf.global_variables_initializer()) 

        ##############################################
        ########## THE AGGREGATION LOOP ##############
        ##############################################

        counter=0
        training_loss_list=[]
        old_loss_list=[]
        new_loss_list=[]
        dataX_new = np.zeros((0,dataX.shape[1]))
        dataY_new = np.zeros((0,dataY.shape[1]))
        dataZ_new = np.zeros((0,dataZ.shape[1]))
        print("dataX dim: ", dataX.shape)

        if(playback_mode):
            controller = ControllerPlayback(dt_steps, state_representation, min_motor_gain, max_motor_gain, frequency_value=frequency_value, stateSize=dataX.shape[1], actionSize=dataY.shape[1])
        else:
            controller = Controller(traj_save_path, save_dir, dt_steps, state_representation, desired_shape_for_traj,
                                left_min, left_max, right_min, right_max, 
                                use_pid_mode=use_pid_mode, frequency_value=frequency_value, stateSize=dataX.shape[1], actionSize=dataY.shape[1], 
                                N=N, horizon=horizon, serial_port=serial_port, baud_rate=baud_rate, DEFAULT_ADDRS=DEFAULT_ADDRS)

        while(counter<num_aggregation_iters):

            print("\n#####################################")
            print("AGGREGATION ITERATION ", counter)
            print("#####################################\n")

            starting_big_loop = time.time()

            print("\n#####################################")
            print("Preprocessing 'new' training data")
            print("#####################################\n")

            dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x)/std_x)
            dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y)/std_y)
            dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z)/std_z)

            ## concatenate state and action, to be used for training dynamics
            inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
            outputs_new = np.copy(dataZ_new_preprocessed)
            print("Done.")

            #################################################
            ### Train dynamics model
            #################################################

            print("\n#####################################")
            print("Training the dynamics model")
            print("#####################################\n")

            training_loss=0
            old_loss=0
            new_loss=0

            if(counter>0):
                training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new, nEpoch, save_dir, fraction_use_new)
            if(counter==0):
                if(train_now):
                    training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new, nEpoch_initial, save_dir, fraction_use_new)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, save_dir+ '/models/model_aggIter' +str(counter)+ '.ckpt')

            #how good is model on training data
            training_loss_list.append(training_loss)
            #how good is model on old dataset
            old_loss_list.append(old_loss)
            #how good is model on new dataset
            new_loss_list.append(new_loss)

            print("\nTraining loss: ", training_loss)

            #####################################
            ## Saving training losses
            #####################################

            np.save(save_dir + '/losses/list_training_loss.npy', training_loss_list) 
            np.save(save_dir + '/losses/list_old_loss.npy', old_loss_list)
            np.save(save_dir + '/losses/list_new_loss.npy', new_loss_list)

            #####################################
            ## Saving model
            #####################################
            if(counter==0):
                saver = tf.train.Saver(max_to_keep=0)
            save_path = saver.save(sess, save_dir+ '/models/model_aggIter' +str(counter)+ '.ckpt')
            print("Model saved at ", save_path)

            #####################################
            ## Validation Metrics
            #####################################

            # print("\n#####################################")
            # print("Calculating Validation Metrics")
            # print("#####################################\n")

            # validation_inputs_states = []
            # labels_1step = []
            # labels_5step = []
            # labels_10step = []
            # labels_15step = []
            # controls_15step=[]
            # max_val_steps = 15

            # #####################################
            # ## make the arrays to pass into forward sim
            # #####################################

            # for i in range(num_rollouts_val):

            #     length_curr_rollout = states_val[i].shape[0]

            #     if(length_curr_rollout>max_val_steps):

            #         #########################
            #         #### STATE INPUTS TO NN
            #         #########################

            #         ## take all except the last max_val_steps pts from each rollout
            #         validation_inputs_states.append(states_val[i][0:length_curr_rollout-max_val_steps])

            #         #########################
            #         #### CONTROL INPUTS TO NN
            #         #########################

            #         #max_val_steps step controls
            #         list_15 = []
            #         for j in range(max_val_steps):
            #             list_15.append(controls_val[i][0+j:length_curr_rollout-max_val_steps+j])
            #             ##for states 0:x, first apply acs 0:x, then apply acs 1:x+1, then apply acs 2:x+2, etc...
            #         list_15=np.array(list_15) #100xstepsx2
            #         list_15= np.swapaxes(list_15,0,1) #stepsx100x2
            #         controls_15step.append(list_15)

            #         #########################
            #         #### STATE LABELS- compare these to the outputs of NN (forward sim)
            #         #########################
            #         labels_1step.append(states_val[i][0+1:length_curr_rollout-max_val_steps+1])
            #         labels_5step.append(states_val[i][0+5:length_curr_rollout-max_val_steps+5])
            #         labels_10step.append(states_val[i][0+10:length_curr_rollout-max_val_steps+10])
            #         labels_15step.append(states_val[i][0+15:length_curr_rollout-max_val_steps+15])

            # validation_inputs_states = np.concatenate(validation_inputs_states)
            # controls_15step = np.concatenate(controls_15step)
            # labels_1step = np.concatenate(labels_1step)
            # labels_5step = np.concatenate(labels_5step)
            # labels_10step = np.concatenate(labels_10step)
            # labels_15step = np.concatenate(labels_15step)

            # #####################################
            # ## pass into forward sim, to make predictions
            # #####################################

            # many_in_parallel=True
            # predicted_15step = dyn_model.do_forward_sim(validation_inputs_states, controls_15step, many_in_parallel, None, None)

            # #####################################
            # ## Calculate validation metrics (mse loss between predicted and true)
            # #####################################

            # array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(labels_1step.shape[0],1))
            # array_stdx = np.tile(np.expand_dims(std_x, axis=0),(labels_1step.shape[0],1))

            # error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_15step[1]-array_meanx,array_stdx)) -np.nan_to_num(np.divide(labels_1step-array_meanx,array_stdx))))
            # error_5step = np.mean(np.square(np.nan_to_num(np.divide(predicted_15step[5]-array_meanx,array_stdx)) -np.nan_to_num(np.divide(labels_5step-array_meanx,array_stdx))))
            # error_10step = np.mean(np.square(np.nan_to_num(np.divide(predicted_15step[10]-array_meanx,array_stdx)) -np.nan_to_num(np.divide(labels_10step-array_meanx,array_stdx))))
            # error_15step = np.mean(np.square(np.nan_to_num(np.divide(predicted_15step[15]-array_meanx,array_stdx)) -np.nan_to_num(np.divide(labels_15step-array_meanx,array_stdx))))

            # print "\n\n", "Multistep error values: ", error_1step, error_5step, error_10step, error_15step

            #####################################
            ## Perform 1 forward simulation
            #####################################

            if(perform_forwardsim_for_vis):
                print("\n#####################################")
                print("Performing a forward simulation of the learned model... using a pre-saved dataset... just for visualization purposes")
                print("#####################################\n")
            
                #for a given set of controls ... compare sim traj vs. learned model's traj (dont expect this to be good cuz error accum)
                many_in_parallel=False
                forwardsim_x_pred = dyn_model.do_forward_sim(forwardsim_x_true, forwardsim_y, many_in_parallel, None, None)    
                forwardsim_x_pred = np.array(forwardsim_x_pred)

                # save results of forward sim
                np.save(save_dir + '/saved_forwardsim/forwardsim_states_true_'+str(counter)+'.npy', forwardsim_x_true)
                np.save(save_dir + '/saved_forwardsim/forwardsim_states_pred_'+str(counter)+'.npy', forwardsim_x_pred)

            #####################################
            ## Run controller for a certain amount of steps
            #####################################

            selected_multiple_u = []
            resulting_multiple_x = []

            for controller_rollout_num in range(num_trajectories_for_aggregation):
                print
                print
                print("PAUSING... right before a controller run... RESET THE ROBOT TO A GOOD LOCATION BEFORE CONTINUING...")
                print
                print
                IPython.embed()
                resulting_x, selected_u, desired_seq = controller.run(num_steps_for_rollout=num_steps_per_controller_run, aggregation_loop_counter=counter, dyn_model=dyn_model)
                selected_multiple_u.append(selected_u)
                resulting_multiple_x.append(resulting_x)

            #np.save(save_dir + '/saved_trajfollow/startingstate_iter' + str(counter) +'.npy', starting_state)
            #np.save(save_dir + '/saved_trajfollow/control_iter' + str(counter) +'.npy', selected_u)
            #np.save(save_dir + '/saved_trajfollow/true_iter' + str(counter) +'.npy', desired_x)
            #np.save(save_dir + '/saved_trajfollow/pred_iter' + str(counter) +'.npy', np.array(resulting_multiple_x))
            
            ### aggregate rollouts into training set
            x_array = np.array(resulting_multiple_x)[0:(rollouts_forTraining+1)] ########the +!???
            u_array = np.array(selected_multiple_u)[0:(rollouts_forTraining+1)] #rollouts x steps x acsize
            for i in range(rollouts_forTraining):
                
                x= x_array[i] #[N+1, NN_inp]
                u= u_array[i][:-1,:] #[N, actionSize]
                
                newDataX= np.copy(x[0:-1, :])
                newDataY= np.copy(u)
                newDataZ= np.copy(x[1:, :]-x[0:-1, :])

                # the actual aggregation
                dataX_new = np.concatenate((dataX_new, newDataX))
                dataY_new = np.concatenate((dataY_new, newDataY))
                dataZ_new = np.concatenate((dataZ_new, newDataZ))

            #####################################
            ## Bookkeeping
            #####################################

            print("\n\nDONE WITH BIG LOOP ITERATION ", counter ,"\n\n")
            print("training dataset size: ", dataX.shape[0] + dataX_new.shape[0])
            print("Time taken: {:0.2f} s\n\n".format(time.time()-starting_big_loop))
            counter= counter+1

        controller.kill_robot()
        return

if __name__ == '__main__':
    main()
