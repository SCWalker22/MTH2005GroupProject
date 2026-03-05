# Question 1e code

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

from svp import svp 

# temperature range
temp = np.arange(250, 330, 0.1)
s = -0.3   # supersaturation changed to 70%
A3 = ((Lv**2 * Rho_w)/(k * Rv * temp**2) + (Rho_w * Rv * temp)/(Kv * svp([0, temp])))**(-1)  # thermodynamic factor
r = 1e-3   # precipitation begins at this droplet size

# rearranging analytic solution (constant of integration is different now since initial droplet size has changed)
# also r = 0 now since we want time taken to evaporate completely
totaltime = (-6.4e-11)/(2*A3*s)

# plotting graph of temperature against time taken
plt.plot(temp, totaltime, color = "cyan")
plt.xlim(250, 330)
plt.xlabel("Temperature (K)")
plt.ylabel("Time taken to evaporate completely (s)")
plt.show()