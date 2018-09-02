from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from decimal import Decimal
import random

#Imports the Data Arc Information
fil_dataArc='Data/PHO_DataArc.csv'
dataArc=open(fil_dataArc)

#For setting print options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

#Aug28_3 best so far
#Aug28_7 best so far
#Sept2_0 is good
fil_hazard_matrix='Data/Hazard_Matrices/Sept2_0.csv'
hazard_matrix=np.genfromtxt(fil_hazard_matrix,delimiter=',')
num_objects,_=hazard_matrix.shape

observed_label=0.1
simulated_label=0.9
pho_label=0.8
threshold=0.99
Sentry_Names=[410777,101955, 29075, 648002, 654427, 99942, 645733, 709909, 608553, 834844]
Sentry_Impact_Chance=[0.0016, 0.00037, 0.00012, 4.7E-05, 3.5E-05, 8.9E-06, 7.7E-06, 6.5E-06, 6.1E-06, 5.8E-06]


#Groups objects according to label
observed_set=[]
simulated_set=[]
pho_set=[]
for i in range(num_objects):
    label=hazard_matrix[i,1]
    name=hazard_matrix[i,0]
    if label==observed_label:
        observed_set.append(hazard_matrix[i])
    elif label==simulated_label:
        simulated_set.append(hazard_matrix[i])
    elif label==pho_label:
        pho_set.append(hazard_matrix[i])
    else:
        print("There's a problem!")
    if int(name) in Sentry_Names:
        print(int(name),Sentry_Names.index(int(name))+1)
        print(hazard_matrix[i,2])
observed_set=np.array(observed_set)
simulated_set=np.array(simulated_set)
pho_set=np.array(pho_set)
num_observed,_=observed_set.shape
num_simulated,_=simulated_set.shape
num_pho,_=pho_set.shape

#Prints some statistics about the objects
ident_obj=[]; unident_obj=[]
num_haz_obs=0; num_haz_sim=0; num_haz_pho=0
for i in range(num_observed):
    if observed_set[i,2]>=threshold:
        num_haz_obs+=1
        ident_obj.append(observed_set[i,0])
    else:
        unident_obj.append(observed_set[i,0])
for i in range(num_simulated):
    if simulated_set[i,2]>=threshold:
        num_haz_sim+=1 
for i in range(num_pho):
    if pho_set[i,2]>=threshold:
        num_haz_pho+=1  
        ident_obj.append(pho_set[i,0])
    else:
        unident_obj.append(pho_set[i,0])      
print('\nTotal objects: %s'%(num_objects))
print('Out of the %s observed objects, %s were identified as hazardous (%.2f%%).'%(num_observed, num_haz_obs, float(num_haz_obs)/float(num_observed)*100.))
print('Out of the %s simulated objects, %s were identified as hazardous (%.2f%%).'%(num_simulated, num_haz_sim, float(num_haz_sim)/float(num_simulated)*100.))
print('Out of the %s PHOs, %s were identified as hazardous (%.2f%%).\n'%(num_pho, num_haz_pho, float(num_haz_pho)/float(num_pho)*100.))

#Prints object to a file
fil_identified_objects='Data/unidentified_objects.txt'
identified_objects=open(fil_identified_objects,'w')
random.shuffle(unident_obj)
for obj in unident_obj:
    identified_objects.write(str(int(obj))+'\n')
identified_objects.close()
print('Printed objects list to %s.'%(fil_identified_objects))

#Finds the OE mean for each class
observed_OE_mean=np.mean(observed_set[:,3:8],axis=0)
simulated_OE_mean=np.mean(simulated_set[:,3:8],axis=0)
pho_OE_mean=np.mean(pho_set[:,3:8],axis=0)
print('The mean OE of observed objects is %s'%(observed_OE_mean))
print('The mean OE of simulated objects is %s'%(simulated_OE_mean))
print('The mean OE of PHO objects is %s'%(pho_OE_mean))

#Finds the OE std for each class
observed_OE_std=np.std(observed_set[:,3:8],axis=0)
simulated_OE_std=np.std(simulated_set[:,3:8],axis=0)
pho_OE_std=np.std(pho_set[:,3:8],axis=0)
print('The STD OE of observed objects is %s'%(observed_OE_std))
print('The STD OE of simulated objects is %s'%(simulated_OE_std))
print('The STD OE of PHO objects is %s'%(pho_OE_std))

##Creates histograms of the hazard ratings
#num_bins=200
#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#ax1.set_title('Observed Objects',fontsize=14)
#ax1.hist(observed_set[:,2], bins=num_bins, color='b', edgecolor='black', linewidth=0.2)
#ax1.set_ylabel('Number of Objects')
#ax2.set_title('Simulated Objects',fontsize=14)
#ax2.hist(simulated_set[:,2], bins=num_bins, color='g', edgecolor='black', linewidth=0.2)
#ax2.set_ylabel('Number of Objects')
#ax3.set_title('Potentially Hazardous Objects',fontsize=14)
#ax3.hist(pho_set[:,2], bins=num_bins, color='r', edgecolor='black', linewidth=0.2)
#ax3.set_ylabel('Number of Objects')
#plt.xlabel('Hazard Rating', fontsize=18)
#plt.show()

##Creates a dict for the data arc
#dataArc_dict={}
#for j,row in enumerate(dataArc):
#    split=row.strip().split(',')
#    data_arc=split[0]
#    if j>0 and len(split)==3 and data_arc!='':
#        name=split[2]
#        dataArc_dict[name]=int(data_arc)

##Sorts the data arcs
#dataArc_found=[]  
#dataArc_unfound=[] 
#unfound_dataArc=0
#for i in range(num_pho):
#    try:
#        if pho_set[i,2]>=threshold:
#            dataArc=dataArc_dict[str(int(pho_set[i,0]))]
#            dataArc_found.append(dataArc)
#        elif pho_set[i,2]<threshold:
#            dataArc=dataArc_dict[str(int(pho_set[i,0]))]
#            dataArc_unfound.append(dataArc)
#        else:
#            print('Something is not right...')
#    except KeyError:
#        unfound_dataArc+=1

#print('%s data arcs where not found...'%(unfound_dataArc))
#num_bins=5000
#f, (ax1, ax2) = plt.subplots(2, sharex=True)
#ax1.set_title('Data Arcs of Found PHOs (Days)')
#ax1.hist(dataArc_found, bins=num_bins, color='b')
#ax2.set_title('Data Arcs of Unfound PHOs (Days)')
#ax2.hist(dataArc_unfound, bins=num_bins, color='g')
#plt.xlim([0,200])
#plt.show()

#Divides the simulated objects into bins according to TuC
num_bins=9
A_Array=np.zeros(num_bins); E_Array=np.zeros(num_bins); IN_Array=np.zeros(num_bins)
N_Array=np.zeros(num_bins); ANGMOM_Array=np.zeros(num_bins); Counter=np.zeros(num_bins)
min_epoch=200
max_epoch=20000
divides=np.linspace(min_epoch,max_epoch,num_bins+1)
for i in range(len(divides)-1):
    for j in range(num_simulated):
        #print(simulated_set[j,0])
        if simulated_set[j,0]>=divides[i] and simulated_set[j,0]<divides[i+1]:
            A_Array[i]+=float(simulated_set[j,3])
            E_Array[i]+=float(simulated_set[j,4])
            IN_Array[i]+=float(simulated_set[j,5])
            N_Array[i]+=float(simulated_set[j,6])
            ANGMOM_Array[i]+=float(simulated_set[j,7])
            Counter[i]+=1.

#Creates x_axis
x_axis=np.zeros(num_bins)
for i in range(num_bins):
    x_axis[i]=(divides[i]+divides[i+1])/2.
num_fit_points=1000
x_fit=np.linspace(min_epoch,max_epoch,num_fit_points)
plt.figure(1)

fSize=20
Mean_As=[]
for i in range(num_bins):
    if A_Array[i]>0:
        Mean_A=A_Array[i]/Counter[i]
        Mean_As.append(Mean_A)
def A_function(x, a, b):
    return a*np.sqrt(x)+b
plt.subplot(221)
ax = plt.gca()
popt_A, pcov_A=curve_fit(A_function, x_axis, Mean_As)
sqrt='$\sqrt{x}$'
a="{:.1E}".format(Decimal(popt_A[0]))
b="{:.1E}".format(Decimal(popt_A[1]))
equation=a+sqrt+'+'+b
plt.text(0.15,0.85,equation, fontsize=12, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
plt.plot(x_fit, A_function(x_fit,*popt_A), color='b')
plt.plot(x_axis,Mean_As,'ro')
plt.xlabel('Time until Collision (years)', fontsize=fSize)
plt.ylabel('Semi-Major Axis (AU)', fontsize=fSize)
plt.title('A Vs. TuC',fontsize=fSize)

Mean_Es=[]
for i in range(num_bins):
    if E_Array[i]>0:
        Mean_E=E_Array[i]/Counter[i]
        Mean_Es.append(Mean_E)
def E_function(x, a, b):
    return a*np.sqrt(x)+b
plt.subplot(222)
ax = plt.gca()
popt_E, pcov_E=curve_fit(E_function, x_axis, Mean_Es)
sqrt='$\sqrt{x}$'
a="{:.1E}".format(Decimal(popt_E[0]))
b="{:.1E}".format(Decimal(popt_E[1]))
equation=a+sqrt+'+'+b
plt.text(0.85,0.85,equation, fontsize=12, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
plt.plot(x_fit, E_function(x_fit,*popt_E), color='b')
plt.plot(x_axis,Mean_Es,'ro')
plt.xlabel('Time until Collision (years)',fontsize=fSize)
plt.ylabel('Eccentricity',fontsize=fSize)
plt.title('E vs. TuC',fontsize=fSize)

Mean_INs=[]
for i in range(num_bins):
    if IN_Array[i]>0:
        Mean_IN=IN_Array[i]/Counter[i]
        Mean_INs.append(Mean_IN)
def IN_function(x, a, b):
    return a*np.sqrt(x)+b
plt.subplot(223)
ax = plt.gca()
popt_IN, pcov_IN=curve_fit(IN_function, x_axis, Mean_INs)
sqrt='$\sqrt{x}$'
a="{:.1E}".format(Decimal(popt_IN[0]))
b="{:.1E}".format(Decimal(popt_IN[1]))
equation=a+sqrt+'+'+b
plt.text(0.15,0.85,equation, fontsize=12, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
plt.plot(x_fit, IN_function(x_fit,*popt_IN), color='b')
plt.plot(x_axis,Mean_INs,'ro')
plt.xlabel('Time until Collision (years)',fontsize=fSize)
plt.ylabel('Inclination (degrees)',fontsize=fSize)
plt.title('I vs. TuC',fontsize=fSize)

Mean_ANGMOMs=[]
conv_ANGMOM=24*3600/1.496e+11
for i in range(num_bins):
    if N_Array[i]>0:
        Mean_ANGMOM=ANGMOM_Array[i]/Counter[i]
        Mean_ANGMOM*=conv_ANGMOM
        Mean_ANGMOMs.append(Mean_ANGMOM)
def ANGMOM_function(x, a, b):
    return a*np.sqrt(x)+b
plt.subplot(224)
ax = plt.gca()
popt_ANGMOM, pcov_ANGMOM=curve_fit(ANGMOM_function, x_axis, Mean_ANGMOMs)
sqrt='$\sqrt{x}$'
a="{:.1E}".format(Decimal(popt_IN[0]))
b="{:.1E}".format(Decimal(popt_IN[1]))
equation=a+sqrt+'+'+b
plt.text(0.15,0.85,equation, fontsize=12, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
plt.plot(x_fit, ANGMOM_function(x_fit,*popt_ANGMOM), color='b')
plt.plot(x_axis,Mean_ANGMOMs,'ro')
plt.xlabel('Time until Collision (years)',fontsize=fSize)
plt.ylabel('Specific Angular Momentum (AU^2/day)',fontsize=fSize)
plt.title('L vs. TuC',fontsize=fSize)
plt.show()

#Mean_Ns=[]
#conv_N=24*3600
#for i in range(num_bins):
#    if N_Array[i]>0:
#        Mean_N=N_Array[i]/Counter[i]
#        Mean_N*=conv_N
#        Mean_Ns.append(Mean_N)
#def N_function(x, a, b):
#    return a*np.sqrt(x)+b
#ax = plt.gca()
#popt_N, pcov_N=curve_fit(N_function, x_axis, Mean_Ns)
#sqrt='$\sqrt{x}$'
#a="{:.1E}".format(Decimal(popt_N[0]))
#b="{:.1E}".format(Decimal(popt_N[1]))
#equation=a+sqrt+'+'+b
#plt.text(0.85,0.85,equation, fontsize=14, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
#plt.plot(x_fit, N_function(x_fit,*popt_N), color='b')
#plt.plot(x_axis,Mean_Ns,'ro')
#plt.xlabel('Time until Collision (years)')
#plt.ylabel('Mean Speed (degrees/Day)')
#plt.title('N vs. TuC')
#plt.show()
#        





 




