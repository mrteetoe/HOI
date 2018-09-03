import numpy as np
from matplotlib import pyplot as plt
import random
import matplotlib as mpl

fil_found='Data/Closeness_Sim/closeness_output.txt'
fil_unfound='Data/Closeness_Sim/closeness_output_unfound.txt'
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

fSize=18
fTitle=20
tickSize=14
Left  = 0.07  # the left side of the subplots of the figure
Right = 0.96    # the right side of the subplots of the figure
Bottom = 0.06   # the bottom of the subplots of the figure
Top = 0.96      # the top of the subplots of the figure
Wspace = 0.11   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
Hspace = 0.3   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
plt.subplots_adjust(left=Left, bottom=Bottom, right=Right, top=Top,
                wspace=Wspace, hspace=Hspace)

num_bins=5000
mpl.rcParams['xtick.labelsize'] = tickSize
mpl.rcParams['ytick.labelsize'] = tickSize
plt.subplot(211)
plt.title('Closest approachs of identified objects (AU)',fontsize=fTitle)
plt.hist(found_dist, bins=num_bins, color='b', edgecolor='black',linewidth=0.2)
plt.ylabel('Number of Objects',fontsize=fSize)
plt.xlabel('Closest Approach of Object (AU)',fontsize=fSize)
plt.xlim([0,2])
plt.ylim([0,700])
plt.subplot(212)
plt.title('Closest approachs of unidentified objects (AU)',fontsize=fTitle)
plt.hist(unfound_dist, bins=num_bins, color='g', edgecolor='black',linewidth=0.2)
plt.ylabel('Number of Objects',fontsize=fSize)
plt.xlabel('Closest Approach of Object (AU)',fontsize=fSize)
plt.xlim([0,2])
plt.ylim([0,700])
plt.show()

print(average_found, std_found)
print(average_unfound, std_unfound)




