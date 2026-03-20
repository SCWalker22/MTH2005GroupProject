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

p = 90000                      # Pressure (900 hPa)
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


# Timestep parameters
dz = 0.1 # Height step (m)
z_top = 500 # Cloud top (m)
dt = dz/W # Find time step
nsteps = int(z_top/dz)


# Fourth order Runge-Kutta

# Initialise arrays
z_rk = np.zeros(nsteps) + 1000# Height (m)
S_rk = np.zeros(nsteps) # Supersaturation
r_rk = np.zeros(nsteps) # Radius of droplet (m)
T_rk = np.zeros(nsteps) # Temperature (K)
P_rk = np.zeros(nsteps) # Pressure (Pa)

# Initial values
S_rk[0] = 0.001 # (0.1% supersaturation), supersaturation 
r_rk[0] = 1e-6 # (1 micron initial roplet radius), radius 
T_rk[0] = T ## temp 
P_rk[0] = p ## pressure 

# RK loop
for i in range(nsteps - 1):
    z_rk[i + 1] = z_rk[i] - dz

    # Get current variables
    s_curr = S_rk[i]
    r_curr = r_rk[i]
    T_curr = T_rk[i]
    P_curr = P_rk[i]

    # k1
    k1s = ds_dt(s_curr, r_curr, T_curr, P_curr)
    k1r = dr_dt(s_curr, r_curr, T_curr, P_curr)
    k1T = dT_dt(s_curr, r_curr, T_curr, P_curr)
    k1P = dP_dt(s_curr, r_curr, T_curr, P_curr)

    # k2
    S2 = s_curr + dt * k1s / 2
    r2 = r_curr + dt * k1r / 2
    T2 = T_curr + dt * k1T / 2
    P2 = P_curr + dt * k1P / 2
    k2s = ds_dt(S2, r2, T2, P2)
    k2r = dr_dt(S2, r2, T2, P2)
    k2T = dT_dt(S2, r2, T2, P2)
    k2P = dP_dt(S2, r2, T2, P2)

    # k3
    S3 = s_curr + dt * k2s / 2
    r3 = r_curr + dt * k2r / 2
    T3 = T_curr + dt * k2T / 2
    P3 = P_curr + dt * k2P / 2
    k3s = ds_dt(S3, r3, T3, P3)
    k3r = dr_dt(S3, r3, T3, P3)
    k3T = dT_dt(S3, r3, T3, P3)
    k3P = dP_dt(S3, r3, T3, P3)

    # k4
    S4 = s_curr + dt * k3s
    r4 = r_curr + dt * k3r
    T4 = T_curr + dt * k3T
    P4 = P_curr + dt * k3P
    k4s = ds_dt(S4, r4, T4, P4)
    k4r = dr_dt(S4, r4, T4, P4)
    k4T = dT_dt(S4, r4, T4, P4)
    k4P = dP_dt(S4, r4, T4, P4)

    # Update
    S_rk[i+1] = s_curr + (dt / 6) * (k1s + 2*k2s + 2*k3s + k4s)
    r_rk[i+1] = r_curr + (dt / 6) * (k1r + 2*k2r + 2*k3r + k4r)
    T_rk[i+1] = T_curr + (dt / 6) * (k1T + 2*k2T + 2*k3T + k4T)
    P_rk[i+1] = P_curr + (dt / 6) * (k1P + 2*k2P + 2*k3P + k4P)

#### plots
    
fig, axes = plt.subplots(2,1, figsize = (8,10),sharex = False)

## first height vs supersation
axes[0].plot(S_rk, z_rk)
axes[0].set_xlabel("Supersaturation")
axes[0].set_ylabel("Height, Meters")
##plt.show()

### Height vs Droplet size
axes[1].plot(r_rk, z_rk)
axes[1].set_xlabel("Droplet Size, Mirco meters")
axes[1].set_ylabel("Height, Meters")
plt.show()

