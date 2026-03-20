"""
D. G. Partridge, ESE, University of Exeter

Program to solve growth of monodisperse cloud droplet population
in an ascending air parcel
"""

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


p = 90000 # let pressure be constant 900HPa
T = 283
X1 = [0,T]
esT = svp(X1)

qv = Ra/Rv * esT/p # water vapor mixing ratio
A1 = g/(Ra*T)*((Lv*Ra)/(c_pa*Rv*T) -1) 
A2 = (Lv**2)/(c_pa*Rv*T**2)+1/qv
A3 = ((Lv**2*Rho_w)/(k*Rv*T**2) + (Rho_w*Rv*T)/(Kv*esT))**(-1)

def droplet_radius(s,r):
    return A3*s/r

def supersaturation(s,r): 
    return (A1*W)-A2*((4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r))


# Defining parameters 
dz = 0.1 # height step (m)
z_top = 2000 # cloud top (m)
dt = dz/W # find time step
nsteps = int(z_top/dz)

z = np.zeros(nsteps) # height 
S = np.zeros(nsteps) # supersaturation
r = np.zeros(nsteps) # droplet radius 

# initial values 
S[0] = 0.001 # initial supersaturation (0.1%)
r[0] = 1e-6 # initial droplet size (1 micron) 


# Forward Euler timestepping scheme 
for i in range(nsteps-1):
    z[i+1]=z[i]+dz
    S[i+1]=S[i]+supersaturation(S[i], r[i])*dt
    r[i+1]=r[i]+droplet_radius(S[i], r[i])*dt
    
    
# plot supersaturation
plt.figure()
plt.plot(S, z)
plt.ylabel("Height (m)")
plt.xlabel("Supersaturation, s")
plt.title("Evolution of Supersaturation in rising parcel", size=12)
plt.grid()
plt.show()


# plot droplet radius
plt.figure()
plt.plot(r * 1e6, z)  # convert meters to microns
plt.ylabel("Height (m)")
plt.xlabel("Droplet Radius (µm)")
plt.title("Evolution of Droplet Radius in rising parcel",size=12)
plt.grid()
plt.show()

# ------------------------------------------------------------------- # 
# Q2b # 
# ------------------------------------------------------------------- #

# Defining parameters 
Dz = [0.1, 2.0] # height step (m)
z_top = 2000 # cloud top (m)

plt.figure(figsize=(10,8))

for j,dz in enumerate(Dz):
    dt = dz/W # find time step
    nsteps = int(z_top/dz)

    z = np.zeros(nsteps) # height 
    S = np.zeros(nsteps) # supersaturation
    r = np.zeros(nsteps) # droplet radius 

    # initial values 
    S[0] = 0.001 # initial supersaturation (0.1%)
    r[0] = 1e-6 # initial droplet size (1 micron) 


    # Forward Euler timestepping scheme 
    for i in range(nsteps-1):
        z[i+1]=z[i]+dz
        S[i+1]=S[i]+supersaturation(S[i], r[i])*dt
        r[i+1]=r[i]+droplet_radius(S[i], r[i])*dt
    

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


    # plotting forward euler vs Rk4 supersaturation
    # plotting forward euler supersaturation
    plt.subplot(2,2,2*j+1)
    plt.plot(S, z, color='blue')
    plt.title(f"Forward Euler (dz={dz})", size = 14)
    plt.ylabel("Height (m)")
    plt.xlabel("Supersaturation (fraction)")
    plt.xlim(0.0008,0.0065)
    plt.grid()

    # plotting rk4 supersaturation
    plt.subplot(2,2,2*j+2)
    plt.plot(Sr, zr, color='red')
    plt.title(f"Runge Kutta 4 (dz={dz})", size = 14)
    plt.ylabel("Height (m)")
    plt.xlabel("Supersaturation (fraction)")
    plt.xlim(0.0008,0.0065)
    plt.grid()


plt.tight_layout()
plt.show()



plt.figure(figsize=(10,8))

for j,dz in enumerate(Dz):
    dt = dz/W # find time step
    nsteps = int(z_top/dz)

    z = np.zeros(nsteps) # height 
    S = np.zeros(nsteps) # supersaturation
    r = np.zeros(nsteps) # droplet radius 

    # initial values 
    S[0] = 0.001 # initial supersaturation (0.1%)
    r[0] = 1e-6 # initial droplet size (1 micron) 


    # Forward Euler timestepping scheme 
    for i in range(nsteps-1):
        z[i+1]=z[i]+dz
        S[i+1]=S[i]+supersaturation(S[i], r[i])*dt
        r[i+1]=r[i]+droplet_radius(S[i], r[i])*dt
    

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

    #print("dz={dz}")
    print(r[-1] * 1e6, rr[-1] * 1e6)
    print(r[-1] * 1e6 - rr[-1] * 1e6)
    
    
    # plotting forward euler vs rk4 droplet growth 
    # plotting forward euler droplet growth 
    plt.subplot(2,2,2*j+1)
    plt.plot(r * 1e6, z, color='blue')
    plt.title(f"Forward Euler (dz={dz})", size = 14)
    plt.ylabel("Height (m)")
    plt.xlabel("Droplet Radius (microns)")
    plt.grid()

    # plotting rk4 droplet growth 
    plt.subplot(2,2,2*j+2)
    plt.plot(rr * 1e6, zr, color='red')
    plt.title(f"Runge Kutta 4 (dz={dz})", size = 14)
    plt.ylabel("Height (m)")
    plt.xlabel("Droplet Radius (microns)")
    plt.grid()


plt.tight_layout()
plt.show()




# ------------------------------------------------------------------- #
# Accuracy test: compare maximum supersaturation for different timesteps
# ------------------------------------------------------------------- #

dz_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.01]   # different vertical steps

euler_max = []
rk4_max = []
difference = []

for dz_test in dz_values:

    dt_test = dz_test / W
    nsteps_test = int(z_top / dz_test)

    # ---------------- Euler ----------------
    S_test = np.zeros(nsteps_test)
    r_test = np.zeros(nsteps_test)

    S_test[0] = 0.001
    r_test[0] = 1e-6

    for i in range(nsteps_test-1):
        S_test[i+1] = S_test[i] + supersaturation(S_test[i], r_test[i]) * dt_test
        r_test[i+1] = r_test[i] + droplet_radius(S_test[i], r_test[i]) * dt_test

    Smax_euler = np.max(S_test)

    # ---------------- RK4 ----------------
    Sr_test = np.zeros(nsteps_test)
    rr_test = np.zeros(nsteps_test)

    Sr_test[0] = 0.001
    rr_test[0] = 1e-6

    for i in range(nsteps_test-1):

        k1s = supersaturation(Sr_test[i], rr_test[i])
        k1r = droplet_radius(Sr_test[i], rr_test[i])

        S2 = Sr_test[i] + dt_test*k1s/2
        r2 = rr_test[i] + dt_test*k1r/2
        k2s = supersaturation(S2, r2)
        k2r = droplet_radius(S2, r2)

        S3 = Sr_test[i] + dt_test*k2s/2
        r3 = rr_test[i] + dt_test*k2r/2
        k3s = supersaturation(S3, r3)
        k3r = droplet_radius(S3, r3)

        S4 = Sr_test[i] + dt_test*k3s
        r4 = rr_test[i] + dt_test*k3r
        k4s = supersaturation(S4, r4)
        k4r = droplet_radius(S4, r4)

        Sr_test[i+1] = Sr_test[i] + (dt_test/6)*(k1s + 2*k2s + 2*k3s + k4s)
        rr_test[i+1] = rr_test[i] + (dt_test/6)*(k1r + 2*k2r + 2*k3r + k4r)

    Smax_rk4 = np.max(Sr_test)

    # store results
    euler_max.append(Smax_euler)
    rk4_max.append(Smax_rk4)
    difference.append(abs(Smax_euler - Smax_rk4))


# convert to %
euler_max = np.array(euler_max) * 100
rk4_max = np.array(rk4_max) * 100
difference = np.array(difference) * 100


# ---------------------------------------------------
# Maximum supersaturation vs timestep
# ---------------------------------------------------

plt.figure()
plt.plot(dz_values, euler_max, marker='o', label="Forward Euler", color = 'blue')
plt.plot(dz_values, rk4_max, marker='o', label="RK4", color = 'red')

plt.xlabel("Height Step dz (m)")
plt.ylabel("Maximum Supersaturation (%)")
plt.title("Maximum Supersaturation vs Model Timestep")
plt.legend()
plt.grid()
plt.show()

