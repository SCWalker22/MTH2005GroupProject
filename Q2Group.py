
import numpy as np
import matplotlib.pyplot as plt
from svp import svp

# --------------------------------------------------------------------------
# Initialise constants (Table A1, Devenish et al., 2016)

Pi = np.pi                     # More precise and cleaner than hardcoding
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


# Define initial vertical velocity and droplet number concentration

# Example placeholders — replace with your values
W = 1                      # Vertical velocity (m/s)
N = 1e8        # Droplet number concentration (#/m^3)

T = 283 #Kelvin
T1 = [0,T]
es = svp(T1)

P = 90000 #Pascals

#defining qv
def qv(P):
    return Ra/Rv * es/ P 

#defining A3 at a given temperature
def A3(T):
    term1 = (Lv)**2 / (k *Rv * (T)**2)
    term2 = (Rv * T) / (Kv * es)
    return 1/ (Rho_w * (term1 + term2))

#defining A1 
def A1(T):
    term1 = g / (Ra*T)
    term2 = (Lv*Ra)/(c_pa * Rv *T) -1
    return term1 * term2

qv = qv(P)

#defining A2
def A2(T):
    #qv = qv(P)
    term1 = Lv**2 / (c_pa * Rv * T**2)
    term2 = 1 / qv
    return term1 + term2

A1 = A1(T)
A2 = A2(T)
A3 = A3(T)

#Supersaturation ODE
def ds_dt(s,r):
    term1 = A1*W
    term2 = A2* ((4* Pi * Rho_w * N)/Rho_a) * r**2 * A3 * (s/r)
    return term1 - term2

#Droplet radius ODE
def dr_dt(s,r):
    return A3 * (s/r)

#-----------------------------------------
# Time stepping 

dz = 0.01           # height step (m)
z_max = 2000.0     # cloud depth (m)

dt = dz / W # time step (s)

steps = int(z_max/dz)

# Arrays to store results
z = np.zeros(steps)
s = np.zeros(steps)
r = np.zeros(steps)

# Initial values
s[0] = 0.0      # initial supersaturation
r[0] = 1e-6       # 1 micron droplet


# FORWARD EULER ----------------

for i in range(steps-1):

    dt = dz / W

    drdt = dr_dt(s[i], r[i])
    dsdt = ds_dt(s[i], r[i])

    r[i+1] = r[i] + dt * drdt
    s[i+1] = s[i] + dt * dsdt

    z[i+1] = z[i] + dz


# PLOTTING ---------------------
plt.plot(z, s)
plt.ylabel("Supersaturation, s",size =13)
plt.xlabel("Height (m)",size =13)
plt.title("Evolution of Supersaturation in rising parcel",size=15)
plt.grid()
plt.show()

plt.plot(z, r * 1e6)
plt.ylabel("Droplet Radius (µm)",size=13)
plt.xlabel("Height (m)",size=13)
plt.title("Evolution of Droplet Radius in rising parcel",size=15)
plt.grid()
plt.show()

