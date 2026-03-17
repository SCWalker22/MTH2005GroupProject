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

def ds_dt(s, r, T_curr, P_curr): #### Supersaturation 
   A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
   dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)
   return (A1*W) - A2 * dql_dt


def dT_dt(s, r, T_curr, P_curr): #### Temperature
    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
    dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)    
    return -(g/c_pa)*W + (Lv/c_pa) * dql_dt


def dP_dt(s, r, T_curr, P_curr):    #### Pressure 
    return (-g * P_curr * W) / (Ra * T_curr)

def terminal_velocity(r: float) -> float:
    """
    
    """
    k_1 = 1.19e6
    k_2 = 8e3
    k_3 = 2.01e3
    if r < 30e-6:
        return k_1*r**2
    if r >= 40e-6 and r < 6e-4:
        return k_2*r
    if r >= 6e-4 and r < 2e-3:
        return k_3*np.sqrt(r)
    return 1

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

def manual_termianl_velodity(r: float) -> float:
    """
    
    """
    t = 0
    dt = 0.01
    t_end = 1
    v = 0
    mass = (4/3)*Rho_w*pi*r**3
    while t < t_end:
        force = down_force(r) - drag(v, r, Rho_a)
        accel = force/mass
        v += dt*accel
        t += dt
    return v


drop_size = np.arange(1e-6, 2e-3, 1e-6)
vels = []
vels_manual = []
for drop in drop_size:
    v = terminal_velocity(drop)
    vels.append(v)
    vels_manual.append(manual_termianl_velodity(drop))
plt.plot(drop_size, vels, label="lecturer")
plt.plot(drop_size, vels_manual, label = "mine")
plt.grid()
plt.legend()
plt.show()