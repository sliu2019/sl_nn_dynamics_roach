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

surfaces = ['carpet', 'styrofoam']
traj_types = ['straight', 'left']
data=[]

for surface in surfaces:

	model_counter = 0
	while(model_counter<2):

		if(surface=='carpet'):
			run_dir = '../run_0'
			if(model_counter==0):
				save_dir=run_dir+'/dd_carpet'
			else:
				save_dir=run_dir+'/a_carpet'
		else:
			run_dir = '../run_2'
			if(model_counter==0):
				save_dir=run_dir+'/dd_foam'
			else:
				save_dir=run_dir+'/b_foam_400pts'

		for traj_type in traj_types:
			cost_per_traj=0

			for run_num in range(10):
				traj_save_path=traj_type+str(run_num)

				if(model_counter==0):
					actions_taken = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_actions.npy')
					desired_states = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_desired.npy')
					traj_taken = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_executed.npy')
					save_perp_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_perp.npy')
					save_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_forward.npy')
					saved_old_forward_dist = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_oldforward.npy')
					save_moved_to_next = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_movedtonext.npy')
					save_desired_heading = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_desheading.npy')
					save_curr_heading = np.load(save_dir +'/'+ traj_save_path +'_diffdrive_currheading.npy')
				else:
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

				cost_per_run_of_traj = np.sum(np.array(list_cost))/actions_taken.shape[0]
				cost_per_traj+= cost_per_run_of_traj

			print(surface, model_counter, traj_type, cost_per_traj/10)
			data.append(cost_per_traj/10)

		model_counter+=1	


#######################################

'''#PLOT
n_groups = 4
means_straight = (int(data[0]), int(data[2]), int(data[4]), int(data[6]))
means_left = (int(data[1]), int(data[3]), int(data[5]), int(data[7]))
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_straight, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Straight')
rects2 = plt.bar(index + bar_width, means_left, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Left')
 
plt.ylabel('Cost')
plt.title('Trajectory Following (cost)')
plt.xticks(index + bar_width, ('DD carpet', 'MB carpet', 'DD styrofoam', 'MB styrofoam'))
plt.legend()
plt.tight_layout()
plt.show()'''

#######################################

#PLOT
#n_groups = 4
DD = (int(data[0]), int(data[1]), int(data[4]), int(data[5]))
MB = (int(data[2]), int(data[3]), int(data[6]), int(data[7]))
fig, ax = plt.subplots()

index = np.array([0,1,3,4])
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, DD, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Diff Drive')
rects2 = plt.bar(index + bar_width, MB, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Model Based')
 
plt.xlabel('CARPET                                              STYROFOAM', fontsize = 14, fontweight='bold', labelpad=20)
plt.title('Trajectory Following (cost)')
plt.xticks(index + bar_width, ('straight', 'left', 'straight', 'left'))
plt.legend()
plt.tight_layout()
plt.show()