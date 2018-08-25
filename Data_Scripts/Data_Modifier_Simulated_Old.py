import numpy as np
from Cart2OE import Cart2OE
import sys
import math

#For setting print options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

dir_inputD='Data/simulated/Dec_4/'
dir_outputD='Data/simulated_new/'

sparsity=50 #How many rows are skipped
num_rows=int(2500/sparsity)
num_objects=20000
num_epochs=20
max_distance=200.
min_distance=0.05

too_far=0
too_close=0
num_unfoundObjects=0
num_convertedObjects=0
outputD=np.zeros((num_rows+1,5))
output_row=np.zeros(5)
prev_num=0
for epoch_num in range(1,num_epochs+1):
    for obj_num in range(1,num_objects+1):
        fil_objectName='object_'+str(obj_num)+'_it'+str(epoch_num*1000)+'.npy'
        try:
            inputD=np.load(dir_inputD+fil_objectName)
            outputD[0]=inputD[0,:5]
            for row_num in range(num_rows):
                r_vec=inputD[row_num*sparsity+1,:3]
                sun_dist=math.sqrt(r_vec[0]**2+r_vec[1]**2+r_vec[2]**2)
                if sun_dist > max_distance:
                    too_far+=1
                    break
                elif sun_dist < min_distance:
                    too_close+=1
                    break
                v_vec=inputD[row_num*sparsity+1,3:6]
                OEs=Cart2OE(r_vec,v_vec)
                output_row[0]=OEs[0]/10. #A in AU (Before division)
                output_row[1]=OEs[1] #EC
                output_row[2]=OEs[2] #IN in radians
                output_row[3]=OEs[5]*1e7 #N in SI (Before multiplication)
                output_row[4]=OEs[6]*1e-16 #ANGMOM in SI (Before multiplication)
                output_row=(output_row)
                outputD[row_num+1,:]=output_row
            np.save(dir_outputD+fil_objectName,outputD)
            num_convertedObjects+=1
        except IOError:
            num_unfoundObjects+=1
        if num_convertedObjects%500==0 and num_convertedObjects>1 and num_convertedObjects!=prev_num:
            print('%s objects converted...'%(num_convertedObjects))
            prev_num=num_convertedObjects
print('Done!')
print('%s objects where found to be too far.'%(too_far))
print('%s objects where found to be too close.'%(too_close))
print('%s objects were not found'%(num_unfoundObjects))


