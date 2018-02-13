import numpy as np
import IPython

def make_trajectory(shape, curr_state, x_index, y_index):

    curr_x = np.copy(curr_state[x_index])
    curr_y = np.copy(curr_state[y_index])
    my_list = []

    if(shape=="straight"):
        i=0
        num_pts = 40
        while(i < num_pts):
            my_list.append(np.array([curr_x+i, curr_y]))
            i+=1

    if(shape=="left"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y]))
        my_list.append(np.array([curr_x+1.8, curr_y+0.2]))
        my_list.append(np.array([curr_x+2, curr_y+1]))
        my_list.append(np.array([curr_x+2, curr_y+2]))
        my_list.append(np.array([curr_x+2, curr_y+3]))
        my_list.append(np.array([curr_x+2, curr_y+4]))
        my_list.append(np.array([curr_x+2, curr_y+5]))
        my_list.append(np.array([curr_x+2, curr_y+6]))
        my_list.append(np.array([curr_x+2, curr_y+7]))

    if(shape=="right"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+0.5, curr_y]))
        my_list.append(np.array([curr_x+0.8, curr_y-0.2]))
        my_list.append(np.array([curr_x+1, curr_y-1]))
        my_list.append(np.array([curr_x+1, curr_y-2]))
        my_list.append(np.array([curr_x+1, curr_y-3]))
        my_list.append(np.array([curr_x+1, curr_y-4]))
        my_list.append(np.array([curr_x+1, curr_y-5]))
        my_list.append(np.array([curr_x+1, curr_y-6]))
        my_list.append(np.array([curr_x+1, curr_y-7]))

    if(shape=="circle_left"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+0.5, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y+0.75]))
        my_list.append(np.array([curr_x+1, curr_y+1.5]))
        my_list.append(np.array([curr_x+0.5, curr_y+2.25]))
        my_list.append(np.array([curr_x, curr_y+2.25]))
        my_list.append(np.array([curr_x-0.5, curr_y+1.5]))
        my_list.append(np.array([curr_x-0.5, curr_y+0.75]))
        my_list.append(np.array([curr_x-0.5, curr_y]))
        my_list.append(np.array([curr_x-0.5, curr_y]))
        my_list.append(np.array([curr_x-0.5, curr_y]))
        my_list.append(np.array([curr_x-0.5, curr_y]))
        my_list.append(np.array([curr_x-0.5, curr_y]))
        my_list.append(np.array([curr_x-0.5, curr_y]))

    if(shape=="zigzag"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y+1]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+3, curr_y+1]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
    '''if(shape=="zigzag"):
        my_list=[]
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+1.5, curr_y-1.5]))
        my_list.append(np.array([curr_x+3, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+5, curr_y]))
        my_list.append(np.array([curr_x+6, curr_y]))
        my_list.append(np.array([curr_x+7, curr_y]))
        my_list.append(np.array([curr_x+8, curr_y]))'''

    if(shape=='figure8'):
        pass

    return np.array(my_list)
