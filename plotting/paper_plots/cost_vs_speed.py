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


traj_paths=['straight', 'circle']

for traj_path in traj_paths:

	list_dd_dist = []
	list_dd_cost = []

	if(traj_path=='straight'):
		save_dir='../run_0'+'/STRAIGHT_cost_vs_speed/saved_straight_dd'
		num = 36
	else:
		save_dir='../run_0'+'/CIRCLE_cost_vs_speed/saved_circle_dd'
		num = 29

	for j in range(num):

		traj_save_path=traj_path+str(j)

		actions_taken = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_actions.npy')
		desired_states = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_desired.npy')
		traj_taken = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_executed.npy')
		save_perp_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_perp.npy')
		save_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_forward.npy')
		saved_old_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_oldforward.npy')
		save_moved_to_next = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_movedtonext.npy')
		save_desired_heading = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_desheading.npy')
		save_curr_heading = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_currheading.npy')

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

		total_cost = np.sum(np.array(list_cost))
		list_dd_dist.append(total_dist)
		list_dd_cost.append(total_cost)

		#print(j, total_cost, total_dist)


	list_mb_dist = []
	list_mb_cost = []

	if(traj_path=='straight'):
		save_dir='../run_0'+'/STRAIGHT_cost_vs_speed/saved_straight_mb_1'
		num = 19
	else:
		save_dir='../run_0'+'/CIRCLE_cost_vs_speed/saved_circle_mb_1'
		num = 21

	for j in range(num):

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

		total_cost = np.sum(np.array(list_cost))
		list_mb_dist.append(total_dist)
		list_mb_cost.append(total_cost)

		#print(j, total_cost, total_dist)


	#make the cost vs. speed plot for the given trajectory
	plt.figure()
	plt.title(traj_path, fontsize = 14, fontweight='bold')
	plt.xlabel('Speed')
	plt.ylabel('Cost')
	plt.plot(list_dd_dist,list_dd_cost,'b.', label='Diff Drive')
	plt.plot(list_mb_dist,list_mb_cost,'r.', label='Model-based')
	plt.plot(np.unique(list_mb_dist), np.poly1d(np.polyfit(list_mb_dist, list_mb_cost, 2))(np.unique(list_mb_dist)), 'r')
	plt.plot(np.unique(list_dd_dist), np.poly1d(np.polyfit(list_dd_dist, list_dd_cost, 2))(np.unique(list_dd_dist)), 'b')
	plt.legend()

plt.show()