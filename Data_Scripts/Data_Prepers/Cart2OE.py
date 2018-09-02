from matplotlib import pyplot as plt
import numpy as np
import math

#For setting printing options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

G=6.67408e-11
M_Sun=1.989e30

#Assumes an input of AU for r_vec
#And km/s for v_vec
#All calculation are done in SI units
#A is outputed in AU
#IN is outputed in degrees
#N and ANGMOM are left in SI units
def Cart2OE(r_vec, v_vec):

    #Does unit conversion
    r_vec*=1.496e+11
    v_vec*=1000.
    
    #Computes the specific angular momentum
    h_bar = np.cross(r_vec,v_vec)
    ANGMOM = np.linalg.norm(h_bar)

    #Computes the radius and velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    #Computes the specific energy and standard gravitational parameter
    mu=G*(M_Sun)
    SE = 0.5*(v**2) - mu/r

    #Computes the semi-major axis
    A = -mu/(2*SE)

    #Computes the eccentricity
    number=(ANGMOM**2)/(A*mu)
    EC = np.sqrt(1.-number)

    #Computes the inclination
    #The abs insures that the degrees are between 0 and 90
    IN = np.arccos(abs(h_bar[2]/ANGMOM))
    if IN > math.pi/2:
        print('Something went wrong')

    #Computes the right ascension of the ascending node
    OM = np.arctan2(h_bar[0],-h_bar[1])
    if OM<0:
        OM=2*math.pi-abs(OM)

    #Computes the latitude of the ascending node
    #beware of division by zero here
    W_TA=np.arctan2(r_vec[2]/math.sin(IN), r_vec[0]*math.cos(OM)+r_vec[1]*math.sin(OM))

    #Computes the true anomaly
    p=A*(1-EC**2)
    TA=np.arctan2(math.sqrt(p/mu)*np.dot(r_vec,v_vec), p-r)
    
    #Argument of periapse
    W=W_TA-TA
    if W<0:
        W=2*math.pi-abs(W)

    #Computes the mean motion
    N = np.sqrt(mu/(A**3))

    A/=1.496e+11 #Changes to AU
    return np.array([A, EC, IN, W, OM, N, ANGMOM])









