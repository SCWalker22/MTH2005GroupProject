"""
D. G. Partridge, ESE, University of Exeter

Program to solve growth of monodisperse cloud droplet population
in an ascending air parcel
"""

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

# --------------------------------------------------------------------------

# %%Advanced: In reality the Thermal Conductivty & Diffusivity are not constant, but depend on the temperature and pressure.
# They also need to include kinetic effects (i.e., the effects of condensation and accommodation coefficients; e.g., Fukuta and Walter 1970).
# Assuming the constant values above is adequate for your project. If you would like to account for the temperature dependance
# please do, you can find further details in the course textbook or in "Lohmann, an introduction to clouds", eq. 7.27, page 192.

# --------------------------------------------------------------------------
# Define initial vertical velocity and droplet number concentration

# Example placeholders — replace with your values
W = None                      # Vertical velocity (m/s)
NDropletDensity = None        # Droplet number concentration (#/m^3)

# --------------------------------------------------------------------------
# Main program starts here

def main():
    """
    Main model driver.
    Implement parcel ascent and droplet growth equations here.
    """
    pass


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 1a #
T = 283
s = 0.003
X1 = [0,T]
esT = svp(X1)
A3 = ((Lv**2*Rho_w)/(k*Rv*T**2) + (Rho_w*Rv*T)/(Kv*esT))**(-1)

def dr(t,r): 
    Dr = A3 * (s/r)
    return Dr

# Defining parameters 
dt = 0.1
te = 40 * 60 # seconds 
r0 = 1e-6
N = int(te/dt)
t = np.linspace(0, te, N)
R_Euler = np.zeros(N)
R_Rk4 = np.zeros(N)

# Forward Euler timestepping scheme 

R_Euler[0]=r0
for n in range(N-1):
    R_Euler[n+1]=R_Euler[n]+dr(t[n], R_Euler[n])*dt

R_Rk4[0]=r0
for n in range(N-1):
    k1 = dr(t[n], R_Rk4[n])
    k2 = dr(t[n] + dt/2, R_Rk4[n] + dt*k1/2)
    k3 = dr(t[n] + dt/2, R_Rk4[n] + dt*k2/2)
    k4 = dr(t[n] + dt, R_Rk4[n] + dt*k3)
    R_Rk4[n+1] = R_Rk4[n] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

r_euler_um = R_Euler * 1e6
r_rk4_um = R_Rk4 * 1e6

plt.figure(figsize=(8,6))
plt.plot(t/60, r_euler_um, label="Forward Euler")
plt.plot(t/60, r_rk4_um, ':', label="RK4", color = 'red')
plt.xlabel("Time (minutes)")
plt.ylabel("Droplet Radius (μm)")
plt.title("Cloud Droplet Growth (T=283K, s=0.30%)")
plt.legend()
plt.grid()
plt.show()