import numpy as np
from svp import svp
import matplotlib.pyplot as plt

# Constants

pi = np.pi                     # More precise and cleaner than hardcoding
g = 9.81                       # Acceleration due to gravity (m s^-2)
c_pa = 1005.0                  # Specific heat capacity of dry air (J kg^-1 K^-1)
Rho_w = 1000.0                 # Density of liquid water (kg m^-3)
Rho_a = 1.225                  # Density of air (kg m^-3)
Eps = 0.622                    # Ratio of molecular masses (water vapour / dry air)
Lv = 2.5e6                     # Latent heat of vaporisation (J kg^-1)
Ra = 287.0                     # Gas constant of dry air (J kg^-1 K^-1)
Rv = 462.0                     # Gas constant of water vapour (J kg^-1 K^-1)
k = 0.024                      # Thermal conductivity of air (J m^-1 s^-1 K^-1)
Kv = 2.21e-5                   # Diffusivity of water vapour (m^2 s^-1)
W = 1.0                        # Vertical velocity (m/s)
N = 1e8                        # Droplet number concentration (#/m^3)


# Initial pressure and temperature

p = 90000                      # Pressure (900 hPa)   ###### google to find better starting vals
T = 283                        # Temperature in Kelvin


# 'p' and 'T' change at each timestep so this calcs SVP, qv, A1, A2 and A3 at each timestep

def calc_coeffs(T_curr, P_curr):
    X1 = [0, T_curr]
    esT = svp(X1)
    qv = Ra/Rv * esT/P_curr
    A1 = g/(Ra*T_curr)*((Lv*Ra)/(c_pa*Rv*T_curr) -1) 
    A2 = (Lv**2)/(c_pa*Rv*T_curr**2)+1/qv
    A3 = ((Lv**2*Rho_w)/(k*Rv*T_curr**2) + (Rho_w*Rv*T_curr)/(Kv*esT))**(-1)
    return A1, A2, A3, esT, qv


# ODEs

def dr_dt(s, r, T_curr, P_curr): #### Droplet radius 
    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
    return A3*s/r

##def ds_dt(s, r, T_curr, P_curr): #### Supersaturation 
##    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
##    dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)
##    return (A1*W) - A2 * dql_dt


def dT_dt(s, r, T_curr, P_curr): #### Temperature
    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
    dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)    
    return -(g/c_pa)*W + (Lv/c_pa) * dql_dt


def dP_dt(s, r, T_curr, P_curr):    #### Pressure 
    return (-g * P_curr * W) / (Ra * T_curr)


#### terminal velocity

def down_force(r: float) -> float:
    """
    Calculate downwards force on a droplet

    Args:
        r: Radius

    Returns:
        Downwards force (N)
    """
    mass: float = Rho_w*pi*r**3
    return mass*g

def drag(
    v: float,
    r: float,
    rho: float

    ) -> float:
    """
    """
    area: float = pi*r**2
    return 0.5*rho*(v**2)*area*0.5

def terminal_velocity(r: float, rho: float) -> float:
    dt = 0.01
    t_end = 1
    t = 0
    v = 0
    while t < t_end:
        mass: float = Rho_w*pi*r**3
        force: float = down_force(r) - drag(v, r, rho)
        accel = force/mass
        v += dt*accel
        t += dt
    return v


#### initial conditions:
init_size_range = np.arange(1e-7, 1e-5,1e-7)
threshold = 1e-9 ### threshold value for our radius to be less than
dt = 0.1
dist_list = []
max_height = 500
s = 0.7

for radius in init_size_range:
    r = radius
    distance = 0
    v = 0
    temp = 283 # kelvin, temp 
    press = 100000# hPa, PRESSURE
    while r > threshold and distance < max_height:
        mass: float = Rho_w*pi*r**3
        force: float = down_force(r) - drag(v, r, Rho_a)
        accel = force/mass
        v += dt*accel
        distance += -dt*v
        # Iterate drop size
        r += dt*dr_dt(s, r, temp, press)

        temp += dt*dT_dt(s, r, temp, press)
        press += dt*dP_dt(s, r, temp, press)
        
    dist_list.append(distance)

## radius size is x, dis_list is y

#### plotting graph:

plt.plot(init_size_range, dist_list)
plt.show()





































