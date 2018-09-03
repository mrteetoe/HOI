import numpy as np
from Cart2OE import Cart2OE
import sys
import math

#For setting print options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

dir_inputD='Data/simulated/simulated_cart/new/'
dir_simulated='Data/simulated/simulated_oe/new/'
fil_simulated_matrix='Data/oe_matrices/so_matrix_NewSept.npy'

sparsity=50 #How many rows are skipped
num_rows=int(2500/sparsity)
num_objects=100
num_epochs=20
max_distance=200.
min_distance=0.05

#too_far=0
#too_close=0
#num_unfoundObjects=0
#num_convertedObjects=0
#outputD=np.zeros((num_rows+1,5))
#output_row=np.zeros(5)
#prev_num=0
#for epoch_num in range(200,20000):
#    for obj_num in range(1,num_objects+1):
#        fil_objectName='SIM'+str(obj_num)+'_LE'+str(epoch_num)+'.npy'
#        try:
#            inputD=np.load(dir_inputD+fil_objectName)
#            inputD=inputD[:2501] #Cutts the matrix in half
#            outputD[0]=inputD[0,:5]
#            for row_num in range(num_rows):
#                r_vec=inputD[row_num*sparsity+1,:3]
#                sun_dist=math.sqrt(r_vec[0]**2+r_vec[1]**2+r_vec[2]**2)
#                if sun_dist > max_distance:
#                    too_far+=1
#                    break
#                elif sun_dist < min_distance:
#                    too_close+=1
#                    break
#                v_vec=-inputD[row_num*sparsity+1,3:6]
#                OEs=Cart2OE(r_vec,v_vec)
#                output_row[0]=OEs[0]/10. #A in AU (Before division)
#                output_row[1]=OEs[1] #EC
#                output_row[2]=OEs[2] #IN in radians
#                output_row[3]=OEs[5]*1e7 #N in SI (Before multiplication)
#                output_row[4]=OEs[6]*1e-16 #ANGMOM in SI (Before multiplication)
#                output_row=(output_row)
#                outputD[row_num+1,:]=output_row
#            np.save(dir_simulated+fil_objectName,outputD)
#            num_convertedObjects+=1
#        except IOError:
#            num_unfoundObjects+=1
#        if num_convertedObjects%500==0 and num_convertedObjects>1 and num_convertedObjects!=prev_num:
#            print('%s objects converted...'%(num_convertedObjects))
#            prev_num=num_convertedObjects
#print('Done!')
#print('%s objects where found to be too far.'%(too_far))
#print('%s objects where found to be too close.'%(too_close))
#print('%s objects were not found'%(num_unfoundObjects))

#Loads simulated and saves to list
so_list=[]
unfound_so=0
found_so=0
zero_matrices=0
last_num=0
nan_found=0
for epoch_num in range(200,20000):
    for object_num in range(1,num_objects+1):
        try:
            fil_name='SIM'+str(object_num)+'_LE'+str(epoch_num)+'.npy'
            so=np.load(dir_simulated+fil_name)
            so=np.flip(so[1:],0) #To take into account the time reversal
            num_origRows,_=so.shape
            if so[-1,0]==0 and so[-1,1]==0:
                zero_matrices+=1
            elif np.isnan(np.sum(so.flatten()))==1:
                nan_found+=1  
            else:
                so_epoch=epoch_num
                so_vector=np.zeros(num_origRows*5+1)
                so_vector[:-1]=so.flatten()
                so_vector[-1]=so_epoch
                so_list.append(so_vector)
                found_so+=1
        except IOError:
            unfound_so+=1
        if found_so%5000==0 and found_so>0 and found_so!=last_num:
            last_num=found_so
            print('%s simulated objects loaded'%(found_so))
print('%s zero matrices were identified'%(zero_matrices))
print('%s vectors with NaN values were found'%(nan_found))
print('%s simulated objects were not found.'%(unfound_so))

#Converts to matrix and saves as numpy array
so_matrix=np.array(so_list)
np.save(fil_simulated_matrix, so_matrix)
print(so_matrix.shape)
print('Simulated matrix saved to %s'%(fil_simulated_matrix))  


