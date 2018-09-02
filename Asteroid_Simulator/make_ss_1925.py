from amuse.lab import *
import numpy as np

data_folder='./'
file_name='cart_planets.csv'

def make_solar_system_1925():
    solar_system = Particles(10)
    planet_matrix = np.genfromtxt(data_folder+file_name, delimiter=",")
    
    sun_position=planet_matrix[-1,2:5] |units.AU
    sun = solar_system[0]
    sun.mass = 1. |units.MSun
    sun.radius = 0 |units.km
    sun.radius_wou = 695700
    sun.position = [0,0,0] |units.AU
    sun.velocity = planet_matrix[-1, 5:8] |units.kms
    sun.name='Sun'

    mercury = solar_system[1]
    mercury.mass = 1.660e-07 |units.MSun
    mercury.radius = 0 |units.km
    mercury.radius_wou = 2440
    mercury.position = planet_matrix[1,2:5] |units.AU
    mercury.position = mercury.position-sun_position
    mercury.velocity = planet_matrix[1,5:8] |units.kms
    mercury.name='Mercury'

    venus = solar_system[2]
    venus.mass = 2.448e-06 |units.MSun
    venus.radius = 0 |units.km
    venus.radius_wou = 6052 
    venus.position = planet_matrix[2,2:5] |units.AU
    venus.position = venus.position-sun_position
    venus.velocity = planet_matrix[2,5:8] |units.kms
    venus.name='Venus'

    earth = solar_system[3]
    earth.mass = 5.972e24 |units.kg
    earth.radius = 0 |units.km
    earth.radius_wou = 6371 
    earth.position = planet_matrix[3,2:5] |units.AU
    earth.position = earth.position-sun_position
    earth.velocity = planet_matrix[3,5:8] |units.kms
    earth.name='Earth'

    mars = solar_system[4]
    mars.mass = 3.227e-07 |units.MSun
    mars.radius = 0 |units.km
    mars.radius_wou= 3390
    mars.position = planet_matrix[4,2:5] |units.AU
    mars.position = mars.position-sun_position
    mars.velocity = planet_matrix[4,5:8] |units.kms
    mars.name='Mars'

    jupiter = solar_system[5]
    jupiter.mass = 9.548e-04 |units.MSun
    jupiter.radius = 0 |units.km
    jupiter.radius_wou = 69911
    jupiter.position = planet_matrix[5,2:5] |units.AU
    jupiter.position = jupiter.position-sun_position
    jupiter.velocity = planet_matrix[5,5:8] |units.kms
    jupiter.name='Jupiter'

    saturn = solar_system[6]
    saturn.mass = 2.859e-04 |units.MSun
    saturn.radius = 0 |units.km
    saturn.radius_wou = 58232
    saturn.position = planet_matrix[6,2:5] |units.AU
    saturn.position = saturn.position-sun_position
    saturn.velocity = planet_matrix[6,5:8] |units.kms
    saturn.name = 'Saturn'

    uranus = solar_system[7]
    uranus.mass = 4.366e-05 |units.MSun
    uranus.radius = 0 |units.km
    uranus.radius_wou = 25362
    uranus.position = planet_matrix[7,2:5] |units.AU
    uranus.position = uranus.position-sun_position
    uranus.velocity = planet_matrix[7,5:8] |units.kms
    uranus.name = 'Uranus'

    neptune = solar_system[8]
    neptune.mass = 5.151e-05 |units.MSun
    neptune.radius = 0 |units.km
    neptune.radius_wou = 24622
    neptune.position = planet_matrix[8,2:5] |units.AU
    neptune.position = neptune.position-sun_position
    neptune.velocity = planet_matrix[8,5:8] |units.kms
    neptune.name = 'Neptune'

    pluto = solar_system[9]
    pluto.mass = 7.396e-09 |units.MSun
    pluto.radius = 0 |units.km
    pluto.radius_wou = 1187 
    pluto.position = planet_matrix[9,2:5] |units.AU
    pluto.position = pluto.position-sun_position
    pluto.velocity = planet_matrix[9,5:8] |units.kms
    pluto.name = 'Pluto'

    return solar_system




