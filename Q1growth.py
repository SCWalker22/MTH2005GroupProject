
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

T = 283       # temperature in K
s = 0.003     # saturation 

r = 1 * 10**-6

t_max = 40 * 60       #duration 
t_step = 0.5      #time step


# Saturation vapour pressure will be provided by your svp() function
# from svp import svp   # Uncomment if svp is in a separate file


#defining A3 at a given temperature
def A3(T):
    es = svp(T)
    term1 = (Lv)**2 / (k *Rv * (T)**2)
    term2 = (Rv * T) / (Kv * es)
    return 1/ (Rho_w * (term1 + term2))

#the value of A, which is constant at temperature T
A3 = A3(T)
               
#growth equation
def dr_dt(r):
    return A3 * (s/r)


#FORWARD EULER--------------

# storing values
t_values = np.arange(0, t_max + t_step, t_step)
r_values = np.zeros(len(t_values))
r_values[0] = r  # initial radius

for n in range(len(t_values) - 1):
    r_values[n+1] = r_values[n] + (t_step * dr_dt(r_values[n]))

#plotting 
plt.plot(t_values, r_values * 10**6)
plt.ylabel("radius, μm")
plt.xlabel("time, seconds")
plt.grid()
plt.show()








