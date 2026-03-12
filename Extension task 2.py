import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Initialise constants (Table A1, Devenish et al., 2016)

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
W = 1.0                      # Vertical velocity (m/s)
N = 1e8        # Droplet number concentration (#/m^3)

# --------------------------------------------------------------------------

p = 90000 # let pressure be constant 900HPa
T = 283
X1 = [0,T]
esT = svp(X1)

dz = 0.1 # height step (m)
z_top = 2000 # cloud top (m)
dt = dz/W # find time step
nsteps = int(z_top/dz)



# liquid water content 
def LWC(N, r): 
    return N*(4/3)*pi*r**3*Rho_w

# liquid water path
def LWP(N, r): 
    lwc = LWC(N,r)
    return np.sum(lwc*dz)

def optical_depth(N, r): 
    lwp = LWP(N, r)
    re = np.mean(r)
    return (3*lwp)/(2*Rho_w*re)

def albedo(N, r): 
    tau = optical_depth(N, r)
    return tau/(tau + 6.7)


qv = Ra/Rv * esT/p # water vapor mixing ratio
A1 = g/(Ra*T)*((Lv*Ra)/(c_pa*Rv*T) -1) 
A2 = (Lv**2)/(c_pa*Rv*T**2)+1/qv
A3 = ((Lv**2*Rho_w)/(k*Rv*T**2) + (Rho_w*Rv*T)/(Kv*esT))**(-1)

def droplet_radius(s,r):
    return A3*s/r

def supersaturation(s,r): 
    return (A1*W)-A2*((4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r))

zr = np.zeros(nsteps) # height 
Sr = np.zeros(nsteps) # supersaturation
rr = np.zeros(nsteps) # droplet radius 

# initial values 
Sr[0] = 0.001 # initial supersaturation (0.1%)
rr[0] = 1e-6 # initial droplet size (1 micron) 

# Fourth order Runge Kutta
for i in range(nsteps-1):
    zr[i+1]=zr[i]+dz

    # k1
    k1s = supersaturation(Sr[i], rr[i])
    k1r = droplet_radius(Sr[i], rr[i])

    # k2
    S2 = Sr[i] + dt*k1s/2
    r2 = rr[i] + dt*k1r/2
    k2s = supersaturation(S2, r2)
    k2r = droplet_radius(S2, r2)

    # k3
    S3 = Sr[i] + dt*k2s/2
    r3 = rr[i] + dt*k2r/2
    k3s = supersaturation(S3, r3)
    k3r = droplet_radius(S3, r3)

    # k4
    S4 = Sr[i] + dt*k3s
    r4 = rr[i] + dt*k3r
    k4s = supersaturation(S4, r4)
    k4r = droplet_radius(S4, r4)

    # update S and r 
    Sr[i+1] = Sr[i] + (dt/6)*(k1s + 2*k2s + 2*k3s + k4s)
    rr[i+1] = rr[i] + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)


r_alb = albedo(N, rr)
plt.figure()
plt.plot(r_alb, zr)
plt.show()