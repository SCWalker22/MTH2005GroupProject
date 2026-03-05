import numpy as np
from svp import svp


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
p = 90000                      # Pressure (900 hPa)
T = 283                        # Temperature in Kelvin


# Calcs SVP and water vapour mixing ratio at 283K

X1 = [0,T]
esT = svp(X1)
qv = Ra/Rv * esT/p


# Calcs constants A1, A2 and A3

A1 = g/(Ra*T)*((Lv*Ra)/(c_pa*Rv*T) -1) 
A2 = (Lv**2)/(c_pa*Rv*T**2)+1/qv
A3 = ((Lv**2*Rho_w)/(k*Rv*T**2) + (Rho_w*Rv*T)/(Kv*esT))**(-1)


# ODEs

def dr_dt(s,r):
    return A3*s/r

def ds_dt(s,r): 
    return (A1*W)-A2*((4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r))


# Timestep parameters 

dz = 0.1 # Height step (m)
z_top = 2000 # Cloud top (m)
dt = dz/W # Find time step
nsteps = int(z_top/dz)


# Forward Euler

# Initialise arrays
z_fe = np.zeros(nsteps)
S_fe = np.zeros(nsteps)
r_fe = np.zeros(nsteps)

# Initial values
S_fe[0] = 0.001
r_fe[0] = 1e-6

# FE loop
for i in range(nsteps - 1):
    z_fe[i+1] = z_fe[i] + dz
    S_fe[i+1] = S_fe[i] + ds_dt(S_fe[i], r_fe[i]) * dt
    r_fe[i+1] = r_fe[i] + dr_dt(S_fe[i], r_fe[i]) * dt


# Fourth order Runge-Kutta

# Initialise arrays
z_rk = np.zeros(nsteps)
S_rk = np.zeros(nsteps)
r_rk = np.zeros(nsteps)

# Initial values
S_rk[0] = 0.001 # (0.1% supersatuartion)
r_rk[0] = 1e-6 # (1 micron initial roplet radius)

# RK loop
for i in range(nsteps - 1):
    z_rk[i + 1] = z_rk[i] + dz

    # k1
    k1s = ds_dt(S_rk[i], r_rk[i])
    k1r = dr_dt(S_rk[i], r_rk[i])

    # k2
    S2 = S_rk[i] + dt * k1s / 2
    r2 = r_rk[i] + dt * k1r / 2
    k2s = ds_dt(S2, r2)
    k2r = dr_dt(S2, r2)

    # k3
    S3 = S_rk[i] + dt * k2s / 2
    r3 = r_rk[i] + dt * k2r / 2
    k3s = ds_dt(S3, r3)
    k3r = dr_dt(S3, r3)

    # k4
    S4 = S_rk[i] + dt * k3s
    r4 = r_rk[i] + dt * k3r
    k4s = ds_dt(S4, r4)
    k4r = dr_dt(S4, r4)

    # Update
    S_rk[i + 1] = S_rk[i] + (dt / 6) * (k1s + 2*k2s + 2*k3s + k4s)
    r_rk[i + 1] = r_rk[i] + (dt / 6) * (k1r + 2*k2r + 2*k3r + k4r)


# Results

print("\nForward Euler:")
print(f"At cloud top (z = {z_top} metres):")
print(f"Supersaturation = {S_fe[-1]}")
print(f"Droplet radius = {r_fe[-1]*1e6} micrometres")

print("\nRunge-Kutta:")
print(f"At cloud top (z = {z_top} metres):")
print(f"Supersaturation = {S_rk[-1]}")
print(f"Droplet radius = {r_rk[-1]*1e6} micrometres")