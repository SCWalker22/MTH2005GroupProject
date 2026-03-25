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

# def down_force(r: float) -> float:
#     """
#     Calculate downwards force on a droplet

#     Args:
#         r: Radius

#     Returns:
#         Downwards force (N)
#     """
#     mass: float = Rho_w*pi*r**3
#     return mass*g

# def drag(
#     v: float,
#     r: float,
#     rho: float

#     ) -> float:
#     """
#     """
#     area: float = pi*r**2
#     return 0.5*rho*(v**2)*area*0.5

# def manual_termianl_velocity(r: float) -> float:
#     """
    
#     """
#     t = 0
#     dt = 0.01
#     t_end = 1
#     v = 0
#     mass = (4/3)*Rho_w*pi*r**3
#     while t < t_end:
#         force = down_force(r) - drag(v, r, Rho_a)
#         accel = force/mass
#         v += dt*accel
#         t += dt
#     return v


# drop_size = np.arange(1e-6, 2e-3, 1e-6)
# vels = []
# vels_manual = []
# for drop in drop_size:
#     v = terminal_velocity(drop)
#     vels.append(v)
#     vels_manual.append(manual_termianl_velodity(drop))
# plt.plot(drop_size, vels, label="lecturer")
# plt.plot(drop_size, vels_manual, label = "mine")
# plt.grid()
# plt.legend()
# plt.show()

def droplet_growth(
    r_current: float,
    r_surrounding: float,
    freq_surrounding: float,

    ) -> float:
    """
    Created a function to calculate the change in drop size due to collision with other drops
    """
    y = 0 # How to calculate horizontal distance
    e = (y**2)/((r_current - r_surrounding)**2)
    pass

def main():
    """
    Main function for Ext4
    """
    droplet_frequency_cm: int = 200 # cm^-3
    droplet_frequency: float = droplet_frequency_cm/(100**3)
    initial_droplet_sizes = np.arange(1e-5, 1e-2, 1e-6) # Are these reasonable sizes???
    r_surrounding = 1e-6 # Is this sensible??
    t = 0
    dt = 0.1
    distance = 1000 # ???
    CLOUD_BASE = 500 # m
    top_radii: list[float] = []
    final_radii: list[float] = []
    total_collision: list[float] = []
    total_coalescence: list[float] = []
    for radius in initial_droplet_sizes:
        collision_this_droplet = 0
        coalescence_this_droplet = 0
        coalescence = 0
        collision = 0
        print(radius)
        r = radius
        while distance > CLOUD_BASE:
            # print(distance)
            vel = terminal_velocity(r)
            collision = dt*droplet_growth(r, r_surrounding, droplet_frequency, vel, W)
            collision_this_droplet += collision
            r += collision
            # Plus supersaturation
            s = 1.05 # Temporary value
            coalescence = dt*dr_dt(s, r, T, p) # S = ???
            r += coalescence
            coalescence_this_droplet += coalescence
            distance -= dt*vel
        total_coalescence.append(coalescence_this_droplet)
        total_collision.append(collision_this_droplet)
        top_radii.append(r)
        while distance >= 0:
            vel = terminal_velocity(r)
            r -= dt*dr_dt(0.7, r, T, p)
            distance -= dt*vel
        final_radii.append(r)
    plt.figure(figsize=(16,9))
    plt.plot(initial_droplet_sizes, final_radii, label="Final Droplet size")
    plt.plot(initial_droplet_sizes, top_radii, label="Radius at top")
    plt.xlabel("Initial Droplet radius", fontsize=14)
    plt.ylabel("Final drop radius", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(16,9))
    plt.plot(initial_droplet_sizes, total_coalescence, label="Growth due to coalescence")
    plt.plot(initial_droplet_sizes, total_collision, label="Growth due to collision")
    plt.xlabel("Initial Droplet radius", fontsize=14)
    plt.ylabel("Final drop radius", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()