# Question 1d code

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
temp = np.arange(275, 330, 0.1)
s = 0.003   # constant supersaturation
A3 = ((Lv**2 * Rho_w)/(k * Rv * temp**2) + (Rho_w * Rv * temp)/(Kv * svp([0, temp])))**(-1)  # thermodynamic factor
r = 1e-3   # raindrop size begins at this droplet size

# rearranging analytic solution (for initial conditions we had in 1a)
# for t to give time taken to reach raindrop size in days
totaltime = (r**2 - 1e-12)/(2*A3*s)  /  (60**2 * 24)

# plotting graph of temperature against time taken
plt.plot(temp, totaltime, color = "purple")
plt.xlim(275, 330)
plt.xlabel("Temperature (K)")
plt.ylabel("Time taken to reach raindrop size (days)")
plt.show()

# printing final time taken to show shortest time
print(totaltime[-1])

# repeating for drizzle size (1x10^-4)
r = 1e-4
totaltime = (r**2 - 1e-12)/(2*A3*s)  /  (60**2)

plt.plot(temp, totaltime, color = "purple")
plt.xlim(275, 330)
plt.xlabel("Temperature (K)")
plt.ylabel("Time taken to reach drizzle size (hours)")
plt.show()

# printing time taken for lowest temperature (longest time)
print(totaltime[0])
