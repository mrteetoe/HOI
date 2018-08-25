from matplotlib import pyplot as plt
import numpy as np

#Imports the Data Arc Information
fil_dataArc='Data/PHO_DataArc.csv'
dataArc=open(fil_dataArc)

#For setting print options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

fil_hazard_matrix='NN/Hazard_Matrices/Aug24_med1.csv'
hazard_matrix=np.genfromtxt(fil_hazard_matrix,delimiter=',')
num_objects,_=hazard_matrix.shape

observed_label=0.1
simulated_label=0.9
pho_label=0.8

#First column is name
#Second column is label
#Third column is hazard rating
#Fourth-eighth are OE means
#Ninth through thirteenth are OE stds

#Groups object according to label
observed_set=[]
simulated_set=[]
pho_set=[]
for i in range(num_objects):
    label=hazard_matrix[i,1]
    if label==observed_label:
        observed_set.append(hazard_matrix[i])
    elif label==simulated_label:
        simulated_set.append(hazard_matrix[i])
    elif label==pho_label:
        pho_set.append(hazard_matrix[i])
    else:
        print("There's a problem!")
observed_set=np.array(observed_set)
simulated_set=np.array(simulated_set)
pho_set=np.array(pho_set)
num_observed,_=observed_set.shape
num_simulated,_=simulated_set.shape
num_pho,_=pho_set.shape

#Finds the OE mean for each class
observed_OE_mean=np.mean(observed_set[:,3:8],axis=0)
simulated_OE_mean=np.mean(simulated_set[:,3:8],axis=0)
pho_OE_mean=np.mean(pho_set[:,3:8],axis=0)
print('The mean OE of observed objects is %s'%(observed_OE_mean))
print('The mean OE of simulated objects is %s'%(simulated_OE_mean))
print('The mean OE of PHO objects is %s'%(pho_OE_mean))

##Creates histograms of the hazard ratings
#num_bins=100
#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#ax1.set_title('Observed Objects')
#ax1.hist(observed_set[:,2], bins=num_bins, color='b')
#ax2.set_title('Simulated Objects')
#ax2.hist(simulated_set[:,2], bins=num_bins, color='g')
#ax3.set_title('Potentially Hazardous Objects')
#ax3.hist(pho_set[:,2], bins=num_bins, color='r')
#plt.show()

#Finds all the different LE used
LEs=list(set(simulated_set[:,0]))

#Gathered statistics according to LE
SO_HR=[]; SO_A=[]; SO_E=[]
SO_IN=[]; SO_N=[]; SO_ANGMOM=[]
for LE in LEs:
    LE=int(LE)
    LE_HR=[LE]
    LE_A=[LE]
    LE_E=[LE]
    LE_IN=[LE]
    LE_N=[LE]
    LE_ANGMOM=[LE]
    for i in range(num_simulated):
        if LE == int(simulated_set[i,0]):
            LE_HR.append(simulated_set[i,2])
            LE_A.append(simulated_set[i,3])
            LE_E.append(simulated_set[i,4])
            LE_IN.append(simulated_set[i,5])
            LE_N.append(simulated_set[i,6])
            LE_ANGMOM.append(simulated_set[i,7])
    SO_HR.append(LE_HR)
    SO_A.append(LE_A) 
    SO_E.append(LE_E)
    SO_IN.append(LE_IN)
    SO_N.append(LE_N)
    SO_ANGMOM.append(LE_ANGMOM)           

##Plots A vs. LE
#x=[]
#y=[]
#for A in SO_A:
#    x.append(A[0])
#    y.append(np.mean(A[1:]))
#plt.scatter(x,y)
#plt.show()

#Plots HR vs. LE
x=[]
y=[]
for HR in SO_HR:
    x.append(HR[0])
    y.append(np.mean(HR[1:]))
plt.scatter(x,y)
plt.show()

#Plots E vs. LE
x=[]
y=[]
for E in SO_E:
    x.append(E[0])
    y.append(np.mean(E[1:]))
plt.scatter(x,y)
plt.show()

#Plots information over the data arcs
dataArc_dict={}
for row in dataArc:
    split=row.strip().split()
    name=split[0]
    data_arc=split[2]
    dataArc_dict[name]=int(data_arc)
    
#for i in range(num_pho):
#    if pho_set[i,2]>=0.5:
        





 




