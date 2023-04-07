import os, hashlib
import requests
from tqdm import tqdm
import numpy as np



def vis_layer_6(pred):
#     map_classes:
#   - drivable_area
#   - ped_crossing
#   - walkway
#   - stop_line
#   - carpark_area
#   - divider
    # pred: shape 6,w,h
    # out: img to be visilized by plt.imshow of shape w,h,3
    try:
        pred = pred.cpu().numpy()
    except:
        pass

    # print("debug:",np.shape(pred))
    w,h = np.shape(pred)[-2],np.shape(pred)[-1]
    color = np.array([(166, 206, 227),(251, 154, 153),(227, 26, 28),(253, 191, 111),(255, 127, 0),(106, 61, 154)])
    array_to_print = np.zeros((np.shape(pred)[1],np.shape(pred)[2],3)) + 255
    thrs = [0.5]*len(pred) 
    # name = ['pedestrians','cars','road_segment', 'lane','drivable_area','ped_crossing','walkway','stop_line','carpark_area']
    order =[0,1,2,3,4,5]
    for layer_idx in order:
        ## init varibles
        cur_pred = pred[layer_idx]
        cur_color = color[layer_idx]
        cur_thr = thrs[layer_idx]
        bin_pred = cur_pred > cur_thr
        
        #clear previous layers if necessary
        clear_mask = (bin_pred == False).astype(int)
        
        cleared_array = array_to_print * np.array([clear_mask,clear_mask,clear_mask]).transpose(1,2,0)
        # add in newthings
        bin_pred = np.array([bin_pred,bin_pred,bin_pred]).transpose(1,2,0)
        array_to_print = cleared_array + bin_pred * cur_color

    # ego car
    # array_to_print[w//2-10:w//2+10,h//2-5:h//2+5,:] = np.array([50,50,100])
    return array_to_print.astype(int)


