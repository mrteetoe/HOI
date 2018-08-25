import numpy as np 

#For setting printing options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

fil_oo='Data/observed_oe/object_732.npy'
#fil_oo='simulated_oe/object_153_it1000.npy'
oo=np.load(fil_oo)

print(oo[0:20])
print(oo.shape)
