
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
W = 10                      # Vertical velocity (m/s)
N = 1e8        # Droplet number concentration (#/m^3)

N_values = np.linspace(1e7, 5e8, 30) #testing what 30 different values of N


# ODEs ---------
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
    return (A1*W)-A2*((4*Pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r))





# Equations for Extension 2 to calculate albedo

def LWC(r, N):
    return N* (4/3) * Pi * (r**3) * Rho_w

def LWP(r, N, dz):
    lwc = LWC(r, N)
    return np.sum(lwc * dz)

def optical_depth(r, N, dz):
    lwp = LWP(r, N, dz)
    r_e = np.mean(r)
    return (3* lwp)/ (2* Rho_w * r_e)

def albedo(r, N, dz):
    tau = optical_depth(r, N, dz)
    return tau / (tau + 6.7)







#Time stepping 

dz = 0.01           # height step (m)
z_max = 2000.0     # cloud depth (m)

dt = dz / W # time step (s)

nsteps = int(z_max/dz)

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




# Liquid water content 
lwc = LWC(rr, N)

# cumulative liquid water path (integral of LWC with height)
LWP_z = np.cumsum(lwc * dz)

#radius 
r_e = rr

# optical depth profile
tau = (3 * LWP_z) / (2 * Rho_w * (r_e + 1e-12)) #plus 1e-12 to avoid dividing by zero

# albedo profile
alpha = tau / (tau + 7.7)


plt.plot(zr, alpha)
plt.xlabel("Height (m)")
plt.ylabel("Cloud Albedo")
plt.title("Cloud Albedo vs Height")
plt.grid()
plt.show()









# now for a changing N ----------

alpha_results = np.zeros(len(N_values))

for j, N in enumerate(N_values):

    # reset arrays
    zr = np.zeros(nsteps)
    Sr = np.zeros(nsteps)
    rr = np.zeros(nsteps)

    Sr[0] = 0.001
    rr[0] = 1e-6

    # Runge–Kutta loop
    for i in range(nsteps-1):

        zr[i+1] = zr[i] + dz

        # k1
        k1s = supersaturation(Sr[i], rr[i])
        k1r = droplet_radius(Sr[i], rr[i])

        #k2
        S2 = Sr[i] + dt*k1s/2
        r2 = rr[i] + dt*k1r/2
        k2s = supersaturation(S2, r2)
        k2r = droplet_radius(S2, r2)

        #k3
        S3 = Sr[i] + dt*k2s/2
        r3 = rr[i] + dt*k2r/2
        k3s = supersaturation(S3, r3)
        k3r = droplet_radius(S3, r3)

        #k4
        S4 = Sr[i] + dt*k3s
        r4 = rr[i] + dt*k3r
        k4s = supersaturation(S4, r4)
        k4r = droplet_radius(S4, r4)

        Sr[i+1] = Sr[i] + (dt/6)*(k1s + 2*k2s + 2*k3s + k4s)
        rr[i+1] = rr[i] + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)

    # compute albedo at cloud top
    lwc = LWC(rr, N)
    LWP_z = np.cumsum(lwc * dz)
    tau = (3 * LWP_z) / (2 * Rho_w * (rr + 1e-12)) #plus 1e-12 to avoid dividing by zero
    alpha = tau / (tau + 7.7)

    alpha_results[j] = alpha[-1]   # cloud-top albedo


plt.plot( N_values, alpha_results)
#plt.xlim(0,1)
plt.xlabel("Droplet concentration, N ")
plt.ylabel("Cloud Albedo")
plt.title("Cloud Albedo vs Droplet Number Concentration")
plt.grid()
plt.show()
















