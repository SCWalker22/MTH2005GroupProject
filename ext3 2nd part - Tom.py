import numpy as np
from svp import svp
import matplotlib.pyplot as plt

# Constants

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
W = 1.0                        # Vertical velocity (m/s)
N = 1e8                        # Droplet number concentration (#/m^3)


# Initial pressure and temperature

p = 90000                      # Pressure (900 hPa)   ###### google to find better starting vals
T = 283                        # Temperature in Kelvin


# 'p' and 'T' change at each timestep so this calcs SVP, qv, A1, A2 and A3 at each timestep

def calc_coeffs(T_curr, P_curr):
    X1 = [0, T_curr]
    esT = svp(X1)
    qv = Ra/Rv * esT/P_curr
    A1 = g/(Ra*T_curr)*((Lv*Ra)/(c_pa*Rv*T_curr) -1) 
    A2 = (Lv**2)/(c_pa*Rv*T_curr**2)+1/qv
    A3 = ((Lv**2*Rho_w)/(k*Rv*T_curr**2) + (Rho_w*Rv*T_curr)/(Kv*esT))**(-1)
    return A1, A2, A3, esT, qv


# ODEs

def dr_dt(s, r, T_curr, P_curr): #### Droplet radius 
    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
    return A3*s/r

##def ds_dt(s, r, T_curr, P_curr): #### Supersaturation 
##    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
##    dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)
##    return (A1*W) - A2 * dql_dt


def dT_dt(s, r, T_curr, P_curr): #### Temperature
    A1, A2, A3, esT, qv = calc_coeffs(T_curr, P_curr)
    dql_dt = (4*pi*Rho_w*N)/Rho_a * r**2 * (A3*s/r)    
    return -(g/c_pa)*W + (Lv/c_pa) * dql_dt


def dP_dt(s, r, T_curr, P_curr):    #### Pressure 
    return (-g * P_curr * W) / (Ra * T_curr)


#### terminal velocity

def down_force(r: float) -> float:
    """
    Calculate downwards force on a droplet

    Args:
        r: Radius

    Returns:
        Downwards force (N)
    """
    mass: float = Rho_w*pi*r**3
    return mass*g

def drag(
    v: float,
    r: float,
    rho: float

    ) -> float:
    """
    Calculate the force of Drag on a droplet of given radius (assuming a spherical droplet)

    Args:
        v: Velocity of droplet
        r: Radius of droplet
        rho: Air pressure

    Returns:
        D: Force of drag
    """
    area: float = pi*r**2
    return 0.5*rho*(v**2)*area*0.5

def manual_terminal_velocity(r: float, rho: float) -> float:
    """
    Calculate terminal velocity of a droplet of given size falling

    Args:
        r: Radius of droplet
        rho: Air pressure

    Returns:
        v: Droplet terminal velocity
    """
    dt = 0.01
    t_end = 1
    t: float = 0
    v: float = 0
    mass: float = Rho_w*pi*r**3
    while t < t_end: # Timestep over a small range to find when droplet stops accelerating - ie at terminal velocity
        force: float = down_force(r) - drag(v, r, rho)
        accel = force/mass
        v += dt*accel
        t += dt
    return v

def terminal_velocity(r: float) -> float:
    """
    Use given formulae to calculate terminal velocity of a droplet at a given radius

    Args:
        r: Radius of drop

    Returns:
        v: Terminal velocity
    """
    k_1 = 1.19e6
    k_2 = 8e3
    k_3 = 2.01e3
    if r < 30e-6:
        return k_1*r**2
    if r >= 40e-6 and r < 6e-4:
        return k_2*r
    if r >= 6e-4 and r < 2e-3:
        return k_3*np.sqrt(r)
    return 1 # If droplet outside of defined range, set terminal velocity to 1

def falling_droplet(size_range: list[float] | np.ndarray[float], velocity_method: str, dt: float = 0.01) -> tuple[list[float | int], list[float | int]]:
    """
    Compute the final size and distance travelled for a range of droplet sizes falling out of a cloud

    Args:
        size_range: List type variable of a range of droplet sizes
        velocity_method: String of which method to use to calculate terminal velocity, "M" or "G"
        dt: Timestep, defaults to 0.01

    Returns:
        dist_list: List of distances fallen before droplet evaporates
        final_drop_size: List of final drop size after falling for 500m (or entirely evaporating)
    """
    #### initial conditions:
    # init_size_range = np.arange(1e-5, 5e-3,1e-5)
    threshold = 1e-7 ### threshold value for our radius to be less than
    dt = 0.01
    dist_list = []
    final_drop_size: list[float] = []
    max_height = 500
    s = 0.7

    # Loop through each drop size
    for radius in size_range:
        r = radius
        distance: float = max_height
        v: float = 0
        temp = T # kelvin, temp 
        press = p# hPa, PRESSURE
        while r > threshold and distance >= 0:
            if velocity_method.upper() == "M":
                mass: float = (4/3)*Rho_w*pi*r**3
                force: float = down_force(r) - drag(v, r, Rho_a)
                accel: float = force/mass
                v += dt*accel
            else: # Use given velocity method
                v = terminal_velocity(r)
            distance -= dt*v
            # Iterate drop size
            # print(dr_dt(s, r, temp, press))
            r -= dt*dr_dt(s, r, temp, press)
        # print(f"{radius=}, {r=}, {distance=}")
        dist_list.append(distance)
        final_drop_size.append(r)
    return dist_list, final_drop_size

def plot_graph(
    x: list[float],
    y: list[float],
    title: str,
    graph_name: str,
    x_label: str = "Droplet size (m)",
    y_label: str = "Height above ground (m)",
    y_lim: None | tuple[int, int] = None,
    y_2: None | list[float] = None

    ):
    """
    Function to create graphs of droplets falling from clous

    Args:
        x: Array like variable of x values
        y: Array like variable of y values
        title: Title of graph to go at top
        graph_name: Filename of graph, saved as Ext3{graph_name}.png
        x_label: Label for x axis, defaults to: Droplet size (m)
        y_label: Label for y axis, defaults to: Height above ground (m)
        y_lim: Limit for y axis in form (y_min, y_max): Defaults to no limit (None)
        y_2: 2nd Set of y values to plot optionally

    Returns:
        None
    """
    plt.figure(figsize=(16,9))
    plt.grid()
    plt.plot(x, y, label="Manual Terminal Velocity")
    if y_2 is not None:
        plt.plot(x, y_2, label="Given Terminal Velocity")
        plt.legend()
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.savefig(f"Ext3{graph_name}.png", dpi=1200)
    plt.show()

def main():
    """
    Main function to produce graphs for Ext3 part 2
    """
    velocity_method = input("Would you like to use the given method for terminal velocity, or our manually calculated one? Please enter G or M respectively: ")
    while velocity_method.upper() not in ["G", "M"]:
        velocity_method = input("Would you like to use the given method for terminal velocity, or our manually calculated one? Please enter G or M respectively: ")
    # Plotting graphs for different size droplets, classified (roughly) into different groups, can clean this up and find actual size ranges
    size_range = np.arange(1e-7, 1e-4,1e-7)
    dist_list, _ = falling_droplet(size_range, velocity_method, dt = 0.001)
    plot_graph(size_range, dist_list, f"Height above ground where droplet 'disappears' (by evaporation)", "Fig1")
    
    size_range = np.arange(1e-5, 3e-3,1e-5)
    dist_list, _ = falling_droplet(size_range, velocity_method, dt = 0.1)
    plot_graph(size_range, dist_list, f"Height above ground where droplet 'disappears' (by evaporation)", "Fig2", y_lim=(0, 500))

    size_range = np.arange(1e-3, 1e-1, 1e-3)
    dist_list, final_sizes = falling_droplet(size_range, velocity_method, dt = 0.1)
    plot_graph(size_range, dist_list, f"Height above ground where droplet 'disappears' (by evaporation)", "Fig3", y_lim=(0, 500))
    plot_graph(size_range, final_sizes, f"Final Droplet radius as droplet evaporates whilst falling", "Fig4", y_label = "Final Droplet Radius (m)")

    size_range = np.arange(1e-2, 1, 1e-2)
    dist_list, final_sizes = falling_droplet(size_range, velocity_method, dt = 0.1)
    plot_graph(size_range, dist_list, f"Height above ground where droplet 'disappears' (by evaporation)", "Fig5", y_lim=(0, 500))
    plot_graph(size_range, final_sizes, f"Final Droplet radius as droplet evaporates whilst falling", "Fig6", y_label = "Final Droplet Radius (m)")
    # Could plot percentage size of initial size???

def multi_plot():
    """
    Plot the difference between the 2 methods of calculating terminal velocity
    """
    # Plotting graphs for different size droplets, classified (roughly) into different groups, can clean this up and find actual size ranges
    size_range = np.arange(1e-7, 1e-4,1e-7)
    dist_list, _ = falling_droplet(size_range, "M", dt = 0.001)
    dist_list_2, _ = falling_droplet(size_range, "G", dt = 0.01)
    plot_graph(size_range, dist_list, "Height above ground where droplet 'disappears' (by evaporation)", "Fig1Multi", y_2 = dist_list_2)
    
    size_range = np.arange(1e-5, 3e-3,1e-5)
    dist_list, _ = falling_droplet(size_range, "M", dt = 0.1)
    dist_list_2, _ = falling_droplet(size_range, "G", dt = 0.1)
    plot_graph(size_range, dist_list, "Height above ground where droplet 'disappears' (by evaporation)", "Fig2Multi", y_lim=(0, 500), y_2 = dist_list_2)

    size_range = np.arange(1e-3, 1e-1, 1e-3)
    dist_list, final_sizes = falling_droplet(size_range, "M", dt = 0.1)
    dist_list_2, final_sizes_2 = falling_droplet(size_range, "G", dt = 0.1)
    plot_graph(size_range, dist_list, "Height above ground where droplet 'disappears' (by evaporation)", "Fig3Multi", y_lim=(0, 500), y_2 = dist_list_2)
    plot_graph(size_range, final_sizes, "Final Droplet radius as droplet evaporates whilst falling", "Fig4Multi", y_label = "Final Droplet Radius (m)", y_2 = final_sizes_2)

    size_range = np.arange(1e-2, 1, 1e-2)
    dist_list, final_sizes = falling_droplet(size_range, "M", dt = 0.1)
    dist_list_2, final_sizes_2 = falling_droplet(size_range, "G", dt = 0.01)
    plot_graph(size_range, dist_list, "Height above ground where droplet 'disappears' (by evaporation)", "Fig5Multi", y_lim=(0, 500), y_2 = dist_list_2)
    plot_graph(size_range, final_sizes, "Final Droplet radius as droplet evaporates whilst falling", "Fig6Multi", y_label = "Final Droplet Radius (m)", y_2 = final_sizes_2)
    # Could plot percentage size of initial size???

if __name__ == "__main__":
    main()
    multi_plot()