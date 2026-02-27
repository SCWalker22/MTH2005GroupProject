# Question1 Code
import numpy as np
import matplotlib.pyplot as plt
from svp import svp

init_size = 1e-6 # 1um
t_end = 40*60 # 40 Mins
s = 0.003 # Supersaturation
T = 283
t_step = 0.001

# Constants
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

def A_fn(
        L_v: float,
        p_w: float,
        k: float,
        R_v: float,
        T: float,
        k_v: float

    ) -> float:
    denom = (L_v**2*p_w)/(k*R_v*T**2) + (p_w*R_v)/(k_v*svp([0, T]))
    return 1/denom

A_3: float = A_fn(Lv, Rho_w, k, Rv, T, Kv)

def drdt(
    A3: float,
    s: float,
    r: float

    ) -> float:
    """
    Function to calculate drdt (change in radius at a specific time)
    Args:
        A3: Thermodynamic Factor (float)
        s: Supersaturation ratio
        r: Radius

    Returns:
        drdt: Change in radius (float)
    """
    return A3*s/r

def forw_euler(r: float) -> list[float]:
    """
    
    """
    r_vals: list[float] = [r]
    t = 0
    while t < t_end:
        prev_r = r_vals[-1]
        r_vals.append(prev_r + t_step*drdt(A_3, s, prev_r))
        t += t_step
    return r_vals

def runge_kutta(r: float) -> float:
    """
    
    """
    r_vals: list[float] = [r]
    t = 0
    while t < t_end:
        prev_r = r_vals[-1]
        k_1 = drdt(A_3, s, prev_r)
        k_2 = drdt(A_3, s, prev_r + 0.5*t_step*k_1)
        k_3 = drdt(A_3, s, prev_r + 0.5*t_step*k_2)
        k_4 = drdt(A_3, s, prev_r + t_step*k_3)
        r_vals.append(prev_r + t_step*(1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4))
        t += t_step
    return r_vals

if __name__ == "__main__":
    forw_vals = forw_euler(init_size)
    rk_vals = runge_kutta(init_size)
    t_vals = np.arange(0, t_end+t_step, t_step)
    plt.figure(figsize=(16,9))
    plt.plot(t_vals, forw_vals, label = "Forward Euler")
    plt.plot(t_vals, rk_vals, label = "Runge_kutta 4")
    plt.legend(loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Droplet size (m)", rotation="horizontal")
    plt.grid()
    plt.show()
    plt.savefig("Q1Stan.png", dpi=1200)