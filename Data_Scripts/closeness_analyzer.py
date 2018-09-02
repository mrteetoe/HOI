import numpy as np
from matplotlib import pyplot as plt
import random

fil_found='Asteroid_Simulator/closeness_output.txt'
fil_unfound='Asteroid_Simulator/closeness_output_unfound.txt'
found=open(fil_found,'r')
unfound=open(fil_unfound,'r')

found_distances=[]
for row in found:
    split=row.strip('\n').split()
    distance=float(split[1])
    found_distances.append(distance)
random.shuffle(found_distances)
found_distances=found_distances[:4400]
found_dist=np.array(found_distances)
print(len(found_distances))

unfound_distances=[]
for row in unfound:
    split=row.strip('\n').split()
    distance=float(split[1])
    unfound_distances.append(distance)
unfound_dist=np.array(unfound_distances)
print(len(unfound_distances))

average_found=np.mean(found_dist)
std_found=np.std(found_dist)
average_unfound=np.mean(unfound_dist)
std_unfound=np.std(unfound_dist)

num_bins=5000
f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title('Closest approachs of identified objects (AU)')
ax1.hist(found_dist, bins=num_bins, color='b', edgecolor='black',linewidth=0.2)
ax1.set_ylabel('Number of Objects')
ax2.set_title('Closest approachs of unidentified objects (AU)')
ax2.hist(unfound_dist, bins=num_bins, color='g', edgecolor='black',linewidth=0.2)
ax2.set_ylabel('Number of Objects')
plt.xlabel('Closest Approach of Object (AU)')
plt.xlim([0,2])
plt.show()

print(average_found, std_found)
print(average_unfound, std_unfound)




