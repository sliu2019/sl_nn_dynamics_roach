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
data=[]

for surface in surfaces:

	model_counter = 0
	while(model_counter<4):

		if(surface=='carpet'):
			run_dir = '../run_0'
			if(model_counter==0):
				save_dir=run_dir+'/dd_carpet'
			elif(model_counter==1):
				save_dir=run_dir+'/a_carpet'
			elif(model_counter==2):
				save_dir=run_dir+'/b_carpet_straight'
			else:
				save_dir=run_dir+'/ab_carpet_straight'
		else:
			run_dir = '../run_2'
			if(model_counter==0):
				save_dir=run_dir+'/dd_foam'
			elif(model_counter==1):
				save_dir=run_dir+'/a_foam_200ptsfromrun0'
			elif(model_counter==2):
				save_dir=run_dir+'/b_foam_400pts'
			else:
				save_dir=run_dir+'/ab_foam_200_400'
		
		cost_per_traj=0
		for run_num in range(10):
			traj_save_path='straight'+str(run_num)

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

		#print(surface, model_counter, cost_per_traj/10)
		data.append(cost_per_traj/10)
		model_counter+=1	


header = ['MODEL', 'Performance on carpet', 'Performance on styrofoam']
print(header)
row = ['Diff Drive', data[0], data[4]]
print(row)
row = ['Model trained on carpet', data[1], data[5]]
print(row)
row = ['Model trained on styrofoam', data[2], data[6]]
print(row)
row = ['Model trained on both', data[3], data[7]]
print(row)


'''<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">MODEL</th>
    <th class="tg-yw4l">Carpet</th>
    <th class="tg-yw4l">Styrofoam</th>
  </tr>
  <tr>
    <td class="tg-yw4l">Diff Drive</td>
    <td class="tg-yw4l">13.85</td>
    <td class="tg-yw4l">15.45</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Model trained on carpet</td>
    <td class="tg-yw4l">5.69</td>
    <td class="tg-yw4l">18.62</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Model trained on styrofoam</td>
    <td class="tg-yw4l">22.25</td>
    <td class="tg-yw4l">8.15</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Model trained on both</td>
    <td class="tg-yw4l">7.52</td>
    <td class="tg-yw4l">15.76</td>
  </tr>
</table>'''