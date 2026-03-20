
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


#effective radius
def eff_rad(N):
    return (LWC/ ((4/3)*Pi*N*Rho_w))**(1/3)


#timestepping---- 

dz = 0.01           # height step (m)
z_max = 2000.0     # cloud depth (m)

dt = dz / W      # time step (s)

nsteps = int(z_max/dz) #step size

#using runge-kutta
def simulate_cloud(N):

    zr = np.zeros(nsteps)
    Sr = np.zeros(nsteps)
    rr = np.zeros(nsteps)

    Sr[0] = 0.001
    rr[0] = 1e-6

    # RK4 solver
    for i in range(nsteps-1):

        zr[i+1] = zr[i] + dz

        k1s = supersaturation(Sr[i], rr[i])
        k1r = droplet_radius(Sr[i], rr[i])

        S2 = Sr[i] + dt*k1s/2
        r2 = rr[i] + dt*k1r/2
        k2s = supersaturation(S2, r2)
        k2r = droplet_radius(S2, r2)

        S3 = Sr[i] + dt*k2s/2
        r3 = rr[i] + dt*k2r/2
        k3s = supersaturation(S3, r3)
        k3r = droplet_radius(S3, r3)

        S4 = Sr[i] + dt*k3s
        r4 = rr[i] + dt*k3r
        k4s = supersaturation(S4, r4)
        k4r = droplet_radius(S4, r4)

        Sr[i+1] = Sr[i] + (dt/6)*(k1s + 2*k2s + 2*k3s + k4s)
        rr[i+1] = rr[i] + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)

    # LWC
    LWC = N * (4/3) * Pi * rr**3 * Rho_w

    # LWP
    LWP_z = np.cumsum(LWC * dz)

    # optical depth
    tau = (3 * LWP_z) / (2 * Rho_w * rr)

    # albedo
    g_asym = 0.85
    alpha = ((1-g_asym)*tau)/(2 + (1-g_asym)*tau)

    return zr, rr, alpha


zr, rr, alpha = simulate_cloud(N)

plt.plot(zr, alpha)
plt.xlabel("Height (m)")
plt.ylabel("Cloud Albedo")
plt.title("Cloud Albedo vs Height")
plt.grid()
plt.show()


# now for a changing N ----------
#without rewriting RK4 for efficiency

alpha_results = np.zeros(len(N_values))

for j, N in enumerate(N_values):

    zr, rr, alpha = simulate_cloud(N)
    
    alpha_results[j] = alpha[-1]   # cloud-top albedo

#plotting N vs albedo
plt.plot( N_values, alpha_results)

plt.xlabel("Droplet concentration, N ")
plt.ylabel("Cloud Albedo")
plt.title("Cloud Albedo vs Droplet Number Concentration")
plt.grid()
plt.show()


