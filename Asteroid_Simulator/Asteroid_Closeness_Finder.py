#Runs simulation of of sun and planet from 2075 t0 2125

from amuse.lab import *
from make_ss_1925 import *
import numpy as np

#Variables to alter
num_asteroids=100
fil_output='closeness_output_unfound.txt'
dt= 0.02 |units.yr
total_t=1000.|units.yr

#For setting printing options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

#Loads names of target asteroids
fil_asteroid_numbers='unidentified_objects.txt'
asteroid_numbers=open(fil_asteroid_numbers)
obj_num=[]
for row in asteroid_numbers:
    num=int(row.strip('\n'))
    obj_num.append(num)

#Loads all asteroids coordinates
fil_asteroid_coord='cart_asteroids.csv'
print('Loading asteroid coordinates...')
loaded_coords=np.genfromtxt(fil_asteroid_coord,delimiter=',')
print('Done!')

starting_row=0
while starting_row<len(obj_num):

    #Resets time and row number
    row_num=0
    t = 0 |units.yr

    #Initialize mercury
    converter = nbody_system.nbody_to_si(1.0 |units.MSun, 1.0 |units.AU)
    mer = Mercury(converter)
    mer.parameters.timestep = 0.05 |units.yr #Sets the initial timestep
    mer.initialize_code()

    #Creates the solar system and adds it to mercury
    ss=make_solar_system_1925()
    mer.particles.add_particles(ss)

    if num_asteroids>(len(obj_num)-starting_row):
        num_asteroids=len(obj_num)-starting_row

    #Find objects and sorts by epoch
    found_obj=[]
    for i,num in enumerate(obj_num[starting_row:starting_row+num_asteroids]):
        found_object=loaded_coords[np.where(loaded_coords[:,0] == num)]
        found_object.resize(found_object.size)
        found_obj.append(found_object)
        
    object_matrix=np.array(found_obj)
    object_matrix=object_matrix[object_matrix[:,1].argsort()]

    #Integrates planets and objects forward
    num_added=0
    asteroid_names=[]
    last_number=0
    x_values=[]; y_values=[]; z_values=[]
    while t < total_t:
        epoch=t.number+2424151.5
        mer.evolve_model(t)
        if t.number<100.:
            while num_added<num_asteroids:
                if object_matrix[row_num,1]>=epoch:
                    new_object = Particles(1)
                    new_object.position= object_matrix[row_num,2:5] |units.AU
                    new_object.velocity= object_matrix[row_num,5:8] |units.kms
                    mer.particles.add_particles(new_object)
                    asteroid_names.append(object_matrix[row_num,0])
                    row_num+=1
                    num_added+=1
                else:
                    break
            t+=dt
        else:
            x_values.append(mer.particles.x.value_in(units.AU))
            y_values.append(mer.particles.y.value_in(units.AU))
            z_values.append(mer.particles.z.value_in(units.AU))
            t+=dt
        if int(t.number%50)==0 and int(t.number)!=last_number:
            print('%s of out of %s years integrated...'%(int(t.number),int(total_t.number)))
            last_number=int(t.number)
    print('Simulation complete!')

    #Closes down mercury
    mer.cleanup_code()
    mer.stop()

    x_matrix=np.array(x_values)
    y_matrix=np.array(y_values)
    z_matrix=np.array(z_values)
    num_steps,_=x_matrix.shape

    #Create a matrix of earth positions
    earth_positions=np.zeros((num_steps,3))       
    earth_positions[:,0]=x_matrix[:,3] 
    earth_positions[:,1]=y_matrix[:,3] 
    earth_positions[:,2]=z_matrix[:,3]

    #Creates matrices for each asteroid
    asteroid_matrices=[]
    num_planets=int(10)
    for i in range(num_asteroids):
        asteroid_positions=np.zeros((num_steps,3))
        asteroid_positions[:,0]=x_matrix[:,i+num_planets] 
        asteroid_positions[:,1]=y_matrix[:,i+num_planets] 
        asteroid_positions[:,2]=z_matrix[:,i+num_planets]
        asteroid_matrices.append(asteroid_positions)

    #Find minimum distance
    print('Appending objects to file %s'%(fil_output))
    output=open(fil_output,'a')
    min_distances=np.zeros((num_asteroids,2))
    for i in range(num_asteroids):
        min_distance=100.
        for j in range(num_steps):
            distance=np.linalg.norm(earth_positions[j]-asteroid_matrices[i][j])
            if distance<min_distance:
                min_distance=distance
        min_distances[i,1]=min_distance
        min_distances[i,0]=asteroid_names[i]
        output.write(str(int(asteroid_names[i]))+" "+str(min_distance)+"\n")
    output.close()
    print('Done!')

    #Increments starting row number
    starting_row+=num_asteroids
    print('\n%s object integrated thus far...\n'%(starting_row))
print('All computations successfully finished!')

    














