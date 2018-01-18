import IPython
import numpy as np 
import matplotlib.pyplot as plt


from moving_distance import moving_distance

#############################################################################

#enter the params for cost function to evaluate on
horiz_penalty_factor= 40
backward_discouragement= 10
heading_penalty_factor= 15

#############################################################################

'''
NOTE: this is cost of full traj... not avg per timestep
'''


traj_paths=['straight', 'left', 'right']

for traj_path in traj_paths:

	save_dir='../run_400'
	total_cost=0
	for j in range(5):

		traj_save_path=traj_path+str(j)

		actions_taken = np.load(save_dir +'/'+ traj_save_path +'_actions.npy')
		desired_states = np.load(save_dir +'/'+ traj_save_path +'_desired.npy')
		traj_taken = np.load(save_dir +'/'+ traj_save_path +'_executed.npy')
		save_perp_dist = np.load(save_dir +'/'+ traj_save_path +'_perp.npy')
		save_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_forward.npy')
		saved_old_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_oldforward.npy')
		save_moved_to_next = np.load(save_dir +'/'+ traj_save_path +'_movedtonext.npy')
		save_desired_heading = np.load(save_dir +'/'+ traj_save_path +'_desheading.npy')
		save_curr_heading = np.load(save_dir +'/'+ traj_save_path +'_currheading.npy')

		#calculate cost
		list_cost = []
		total_dist = 0
		for i in range(actions_taken.shape[0]):
			p = save_perp_dist[i]
			ND = save_forward_dist[i]
			OD = saved_old_forward_dist[i]
			moved_to_next = save_moved_to_next[i]
			a = save_desired_heading[i]
			h = save_curr_heading[i]
			diff = np.abs(moving_distance(a, h))

			#write this as desired
			cost = p*horiz_penalty_factor
			cost += diff*heading_penalty_factor
			if(moved_to_next==0):
				cost += (OD - ND)*backward_discouragement

			list_cost.append(cost)
			if(i==0):
				total_dist=0
			else:
				x_diff = traj_taken[i][0]-traj_taken[i-1][0]
				y_diff = traj_taken[i][1]-traj_taken[i-1][1]
				total_dist+= np.sqrt(x_diff*x_diff + y_diff*y_diff)

		avg_cost_of_run = np.sum(np.array(list_cost))/actions_taken.shape[0]
		total_cost+= avg_cost_of_run
	total_cost/=5

	print(traj_path, total_cost)