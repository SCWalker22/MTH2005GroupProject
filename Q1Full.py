# Question 1 Combined
# All code for Q1 is in here, in order, there are some shared values, the constants at the top
# But other functionality should all be added to the function, ie in def part_a():

# ============================= Please go through and check you are happy with your section, and comment====================================

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from svp import svp

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
init_size = 1e-6               # Initial droplet size (m)
t_end = 40*60                  # Time of iterations (s)
s = 0.003                      # Supersaturation (%)
T = 283                        # Atmospheric Temperature (K)
t_step = 0.001                 # Time step (s)

def A_fn(
        L_v: float,
        p_w: float,
        k: float,
        R_v: float,
        T: float | list[float],
        k_v: float

    ) -> float | list[float]:
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
        A_3: Thermodynamic factor as defined in notes (Devenish et al., 2016)
    """
    denom = (L_v**2*p_w)/(k*R_v*T**2) + (p_w*R_v)/(k_v*svp([0, T]))
    return denom**(-1)

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
    """
    t_vals = np.arange(0, t_end+t_step, t_step)
    A_3: float = A_fn(Lv, Rho_w, k, Rv, T, Kv)
    # Forward Euler
    r_euler = forw_euler(init_size, A_3, t_step, t_end=t_end)

    # Runge-Kutta
    r_rk4 = runge_kutta(init_size, A_3, t_step, t_end=t_end)

    plt.figure(figsize=(16,9))
    plt.plot(t_vals/60, r_euler, label="Forward Euler")
    plt.plot(t_vals/60, r_rk4, ':', label="RK4", color = 'red')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Droplet Radius (μm)")
    plt.title("Cloud Droplet Growth (T=283K, s=0.30%)")
    plt.legend()
    plt.grid()
    plt.savefig("Q1A.png", dpi=1200)
    plt.show()

def graph_slices(
        end_time: int = 40*60,
        num_steps: int = 10,
        t_step: int = 0.001
    ) -> None:
    """
    Create a Graph from a 2D slice of temperature, initial radius, final radius graph, based on user input in CLI

    Args:
        end_time: (Optional) end of iteration time in seconds
        num_steps: (Option) Number of computation steps on x-axis (number of points)
        t_step: (Optional) Time delta between iteration steps
    """
    temp_or_size: str = input("Would you like to vary temperature or size? (T, r): ")
    while temp_or_size.lower() not in ["t", "r", "exit", "quit", "q", ""]:
        temp_or_size: str = input("Would you like to vary temperature or size? (T, r): ")
    
    if temp_or_size.lower() in ["exit", "quit", "q", ""]:
        # Allow user to exit loop early if needed
        return None

    final_sizes: list[float] = []
    if temp_or_size == "T":
        graph_title: str = f"Plot of changing temperature on growth of rain drop after {t_end/60} minutes"
        x_axis_label: str = "Temperature (K)"
        min_temp: int = int(input("Enter minimum temperature (K): "))
        max_temp: int = int(input("Enter maximum temperature (K): "))
        init_size: float = float(input("Enter initial radius (m - Try 1e-6): "))
        other_val = init_size
        temperatures = np.linspace(min_temp, max_temp, num_steps+1)
        x = temperatures
        for temp in temperatures:
            print(temp)
            A_3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv)
            final_size = runge_kutta(init_size, A_3, t_step, t_end=end_time)[-1]
            final_sizes.append(final_size)

    else:
        graph_title: str = f"Plot of changing initial radius on growth of rain drop after {t_end/60} minutes"
        x_axis_label: str = "Initial Radius (m)"
        min_radius: float = float(input("Enter minimum radius (m): "))
        max_radius: float = float(input("Enter maximum radius (m): "))
        temp: int = int(input("Enter temperature (K - Try 283): "))
        other_val = temp
        radii = np.linspace(min_radius, max_radius, num_steps+1)
        x = radii
        A_3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv)
        for radius in radii:
            final_size = runge_kutta(radius, A_3, t_step, t_end=end_time)[-1]
            final_sizes.append(final_size)

    plt.figure(figsize=(16, 9))
    plt.grid()
    plt.title(graph_title)
    plt.xlabel(x_axis_label)
    plt.ylabel("Final size of drop (m)")
    print(f"{len(x)=}, {len(final_sizes)=}")
    plt.plot(x, final_sizes)
    plt.savefig(f"{temp_or_size} [{min(x)}, {max(x)}] @ {other_val}")
    plt.show()


def part_c():
    """
    Code to compute Q1 Part C
    """
    # Set up plot to be 3D
    plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")
    temp_range = np.arange(200, 350, 10) # Create a range of temperatures to iterate through
    init_size_range = np.arange(1e-7, 1e-5, 1e-7) # Create a range of intial sizes to loop through
    times: list[int] = [10*60, 20*60, 30*60, 40*60] # List of times to loop thorugh, to check growth throughout time frame
    X, Y = np.meshgrid(init_size_range, temp_range) # Create a meshgrid for X, Y axis, allows us to do a 3D plot
    for time_end in times: # Loop through times
        final_sizes: list[list[float]] = [] # Create an array to fill with final sizes, it is a 2D array since we have a 3D graph
        for temp in temp_range: # Loop through different temperaturs
            sizes: list[float] = []
            A_3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv) # Calculate the thermodynamic factor for each temperature
            for init_size in init_size_range: # Loop through each intial size
                final_size = runge_kutta(init_size, A_3, 1, t_end=time_end)[-1] # Calculate the final size (uses RK scheme - Faster w/ FE?)
                sizes.append(final_size)
            final_sizes.append(sizes) # Append the final sizes to our list
        # X, Y = np.meshgrid(init_size_range, temp_range)
        # ax = plt.axes(projection="3d")
        # Set up graph
        ax.set_title("3D surface of drop size by varying temperature and initial drop size")
        ax.set_xlabel("Initial Drop size (m)")
        ax.set_ylabel("Temperature")
        ax.plot_surface(X, Y, np.array(final_sizes), label=f"Drop Size after {time_end} mins")
    plt.savefig("Q1C(3D).png", dpi=1200)
    plt.show()

    # Attempting to plot some 2D slices of graph
    for _ in range(5):
        graph_slices()

def part_d():
    """
    """
    # temperature range
    temp = np.arange(250, 330, 0.1)
    s = 0.003   # constant supersaturation
    A3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv)
    r = 1e-3   # precipitation begins at this droplet size

    # rearranging analytic solution (for initial conditions we had in 1a)
    # for t to give time taken to reach precipitation size in days
    totaltime = (r**2 - 1e-12)/(2*A3*s)  /  (60**2 * 24)

    # plotting graph of temperature against time taken
    plt.figure(figsize=(16, 9))
    plt.plot(temp, totaltime, color = "purple")
    plt.xlim(250, 330)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Time taken to reach precipitation size (days)")
    plt.savefig("Q1D.png", dpi=1200)
    plt.show()

    # printing final time taken to show shortest time
    print(totaltime[-1])

def part_e():
    """
    """
    # temperature range
    temp = np.arange(250, 330, 0.1)
    s = -0.3   # supersaturation changed to 70%
    A3 = A_fn(Lv, Rho_w, k, Rv, temp, Kv)
    r = 1e-3   # precipitation begins at this droplet size

    # rearranging analytic solution (constant of integration is different now since initial droplet size has changed)
    # also r = 0 now since we want time taken to evaporate completely
    totaltime = (-6.4e-11)/(2*A3*s)

    # plotting graph of temperature against time taken
    plt.figure(figsize=(16, 9))
    plt.plot(temp, totaltime, color = "cyan")
    plt.xlim(250, 330)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Time taken to evaporate completely (s)")
    plt.savefig("Q1E.png", dpi=1200)
    plt.show()


if __name__ == "__main__":
    part_a()
    # No code needed for part b
    part_c()
    part_d()
    part_e()