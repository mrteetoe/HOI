import numpy as np

#For setting printing options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

#Variables to alter
num_origRows=50

#Output files
fil_observed_matrix='Data/oo_matrixN.npy'
fil_simulated_matrix='Data/so_matrixN.npy'
fil_pho_matrix='Data/pho_matrixN.npy'

##Loads observed and saves to list
#dir_observed='Data/pho_new/'
#max_num=int(1e6)
#oo_list=[]
#unfound_oo=0
#last_num=0
#nan_found=0
#for object_num in range(1,max_num+1):
#    try:
#        fil_name='object_'+str(object_num)+'.npy'
#        oo=np.load(dir_observed+fil_name)
#        if np.isnan(np.sum(oo[1:].flatten()))==1:
#            nan_found+=1
#        else:  
#            oo_num=oo[0,0]
#            oo_vector=np.zeros(num_origRows*5+1)
#            oo_vector[:-1]=oo[1:].flatten()
#            oo_vector[-1]=oo_num
#            oo_list.append(oo_vector)
#    except IOError:
#        unfound_oo+=1
#    if object_num%5000==0 and object_num>0 and object_num!=last_num:
#        last_num=object_num
#        print('%s observed objects loaded'%(object_num))
#print('%s vectors with NaN values were found'%(nan_found))
#print('%s observed objects were not found.'%(unfound_oo))

##Converts to matrix and saves as numpy array
#oo_matrix=np.array(oo_list)
#np.save(fil_pho_matrix, oo_matrix)
#print('Observed matrix saved to %s'%(fil_observed_matrix))

##Loads simulated and saves to list
#dir_simulated='Data/simulated_oe/'
#num_epochs=20
#num_objects=20000
#so_list=[]
#unfound_so=0
#found_so=0
#zero_matrices=0
#last_num=0
#nan_found=0
#for epoch_num in range(1,num_epochs+1):
#    for object_num in range(1,num_objects+1):
#        try:
#            fil_name='object_'+str(object_num)+'_it'+str(epoch_num*1000)+'.npy'
#            so=np.load(dir_simulated+fil_name)
#            if so[-1,0]==0 and so[-1,1]==0:
#                zero_matrices+=1
#            elif np.isnan(np.sum(so[1:].flatten()))==1:
#                nan_found+=1  
#            else:
#                so_epoch=epoch_num*1000
#                so_vector=np.zeros(num_origRows*5+1)
#                so_vector[:-1]=so[1:].flatten()
#                so_vector[-1]=so_epoch
#                so_list.append(so_vector)
#                found_so+=1
#        except IOError:
#            unfound_so+=1
#        if found_so%5000==0 and found_so>0 and found_so!=last_num:
#            last_num=found_so
#            print('%s simulated objects loaded'%(found_so))
#print('%s zero matrices were identified'%(zero_matrices))
#print('%s vectors with NaN values were found'%(nan_found))
#print('%s simulated objects were not found.'%(unfound_so))

##Converts to matrix and saves as numpy array
#so_matrix=np.array(so_list)
#np.save(fil_simulated_matrix, so_matrix)
#print(so_matrix.shape)
#print('Simulated matrix saved to %s'%(fil_simulated_matrix))  

##Inspects matrix
#test_matrix=np.load(fil_simulated_matrix)  
#print(test_matrix.shape)
#print(test_matrix[0,-5:])   

#Loads simulated and saves to list
dir_simulated='Data/simulated_oe/'
#num_epochs=20
num_objects=100
so_list=[]
unfound_so=0
found_so=0
zero_matrices=0
last_num=0
nan_found=0
for epoch_num in range(20,20000):
    for object_num in range(1,num_objects+1):
        try:
            fil_name='SIM'+str(object_num)+'_LE'+str(epoch_num)+'.npy'
            so=np.load(dir_simulated+fil_name)
            if so[-1,0]==0 and so[-1,1]==0:
                zero_matrices+=1
            elif np.isnan(np.sum(so[1:].flatten()))==1:
                nan_found+=1  
            else:
                so_epoch=epoch_num*1000
                so_vector=np.zeros(num_origRows*5+1)
                so_vector[:-1]=so[1:].flatten()
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












