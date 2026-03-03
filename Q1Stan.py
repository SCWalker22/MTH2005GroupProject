# Question1 Code
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from svp import svp

# Constants for Q 1 (a)
init_size = 1e-6 # 1um
t_end = 40*60 # 40 Mins in seconds
s = 0.003 # Supersaturation
T = 283 # Temperature (K)
t_step = 0.001 # Time step (s)

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
    """
    Compute A_3 using provided SVP (Satruation Vapour Pressure) function, and constants, including temperature

    Args:
        L_v: Latent heat of vaporisation
        p_w: Density of water
        k: Thermal conductivity of air
        R_v: Water vapor gas constant
        T: Temperature (Kelvin)
        k_v: Diffusivity of Water in air

    Returns:
        A_3: Thermodynamic paper as defined in notes
    """
    denom = (L_v**2*p_w)/(k*R_v*T**2) + (p_w*R_v)/(k_v*svp([0, T]))
    return 1/denom

A_3_part_1: float = A_fn(Lv, Rho_w, k, Rv, T, Kv)

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

def forw_euler(
    r: float,
    A_3: float,
    t_step: float,
    t_end: int = 40*60
    
    ) -> list[float]:
    """
    Compute forward euler iteration for a given droplet size, thermodynamic factor, and over a given time interval

    Args:
        r: Initial radius of drop (m)
        A_3: Thermodynamic factor (m^2s^{-1})
        t_step: Time step between iterations (s)
        t_end: (Optional - Defualt 40 mins) Final time (starting at t=0) (s)

    Returns:
        List of drop sizes at each timestep
    """
    r_vals: list[float] = [r]
    t = 0
    while t < t_end: # Loop through each timestep
        prev_r = r_vals[-1]
        r_vals.append(prev_r + t_step*drdt(A_3, s, prev_r)) # Calculate next drop size using forward Euler
        t += t_step
    return r_vals

def runge_kutta(
    r: float,
    A_3: float,
    t_step: float,
    t_end: int = 40*60 # 40 mins
    
    ) -> float:
    """
    4th Order Runge-Kutta iteration for a given droplet size, thermodynamic factor, and over a given time interval

    Args:
        r: Initial radius of drop (m)
        A_3: Thermodynamic factor (m^2s^{-1})
        t_step: Time step between iterations (s)
        t_end: (Optional - Defualt 40 mins) Final time (starting at t=0) (s)
    
    Returns:
        List of drop size at each time step
    """
    r_vals: list[float] = [r]
    t = 0
    while t < t_end: # Loop through each time step
        prev_r = r_vals[-1]
        # Calculaate all 4 k vals
        k_1 = drdt(A_3, s, prev_r)
        k_2 = drdt(A_3, s, prev_r + 0.5*t_step*k_1)
        k_3 = drdt(A_3, s, prev_r + 0.5*t_step*k_2)
        k_4 = drdt(A_3, s, prev_r + t_step*k_3)
        r_vals.append(prev_r + t_step*(1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)) # Append next drop size as found by Runge-Kutta
        t += t_step
    return r_vals

def part_a():
    """
    Code to compute Q1 Part a
    """
    forw_vals = forw_euler(init_size, A_3_part_1, t_step)
    rk_vals = runge_kutta(init_size, A_3_part_1, t_step)
    t_vals = np.arange(0, t_end+t_step, t_step)
    plt.figure(figsize=(16,9))
    plt.plot(t_vals, forw_vals, label = "Forward Euler")
    plt.plot(t_vals, rk_vals, label = "Runge_kutta 4")
    plt.legend(loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Droplet size (m)", rotation="horizontal")
    plt.grid()
    plt.show()

def part_c():
    """
    Code to compute Q1 Part C
    """
    ax = plt.axes(projection="3d")
    temp_range = np.arange(200, 350, 10) # 100 to 350K in 1K increments
    init_size_range = np.arange(1e-7, 1e-5, 1e-7)
    times: list[int] = [10*60, 20*60, 30*60, 40*60]
    X, Y = np.meshgrid(init_size_range, temp_range)
    # for time_end in times:
    #     final_sizes: list[list[float]] = []
    #     for temp in temp_range:
    #         sizes: list[float] = []
    #         A_3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv)
    #         for init_size in init_size_range:
    #             final_size = runge_kutta(init_size, A_3, 1, t_end=time_end)[-1]
    #             sizes.append(final_size)
    #         final_sizes.append(sizes)
    #     # X, Y = np.meshgrid(init_size_range, temp_range)
    #     # ax = plt.axes(projection="3d")
    #     ax.set_title("3D surface of drop size by varying temperature and initial drop size")
    #     ax.set_xlabel("Initial Drop size (m)")
    #     ax.set_ylabel("Temperature")
    #     ax.plot_surface(X, Y, np.array(final_sizes), label=f"Drop Size after {time_end} mins")
    #     # Can plot multiple surafces every 10 mins???
    # plt.show()

    final_sizes: list[float] = []
    A_3 = A_fn(Lv, Rho_w, k, Rv, 283, Kv)
    for init_size in init_size_range:
        final_size = runge_kutta(init_size, A_3, 0.1)[-1]
        final_sizes.append(final_sizes)
    plt.plot(init_size_range, final_sizes)
    plt.show()


if __name__ == "__main__":
    # part_a()
    part_c()