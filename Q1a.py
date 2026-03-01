import numpy as np
import matplotlib.pyplot as plt

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

# Saturation vapour pressure will be provided by your svp() function
from svp import svp   # Uncomment if svp is in a separate file

# Question 1 code

# defining constants
temp = 283   # temperature (constant to begin with)
s = 0.003   # constant supersaturation
A3 = ((Lv**2 * Rho_w)/(k * Rv * temp**2) + (Rho_w * Rv * temp)/(Kv * svp([0, temp])))**(-1)

# equation for dr/dt 
def dr(r):
    return A3*(s/r)

# initial conditions + timestep
rvalues = np.zeros(4801)
rvalues[0] = 1e-6      # 1 micron (10^-6 metres)
t = np.linspace(0, 2400, 4801)
dt = 0.5

# Forward Euler loop
for i in range(4800):
    r = rvalues[i]
    rnew = r + dt * dr(r)
    rvalues[i+1] = rnew

plt.plot(t, rvalues, color = "black")
plt.xlim(0, 2400)
plt.show()

# re-initialising conditions for RK4
rvalues = np.zeros(4801)
rvalues[0] = 1e-6

# RK4 loop
for i in range(4800):
    r = rvalues[i]
    k1 = dr(r)
    k2 = dr(r + 0.5*dt*k1)
    k3 = dr(r + 0.5*dt*k2)
    k4 = dr(r + dt*k3)

    rnew = r + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    rvalues[i+1] = rnew

plt.plot(t, rvalues, color = "black")
plt.xlim(0, 2400)
plt.show()