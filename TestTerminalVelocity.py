import numpy as np
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
    """
    area: float = pi*r**2
    return 0.5*rho*(v**2)*area*0.5

def manual_terminal_velocity(r: float, rho: float) -> float:
    """
    
    """
    dt = 0.01
    t_end = 1
    t = 0
    v = 0
    while t < t_end:
        mass: float = Rho_w*pi*r**3
        force: float = down_force(r) - drag(v, r, rho)
        accel = force/mass
        v += dt*accel
        t += dt
    return v

def terminal_velocity(r: float) -> float:
    """
    
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
    return 1

if __name__=="__main__":
    r_range = np.arange(1e-6, 2e-3, 1e-6)
    vels_given: list[float] = []
    vels_manual: list[float] = []
    for r in r_range:
        vels_given.append(terminal_velocity(r))
        vels_manual.append(manual_terminal_velocity(r, Rho_a))
    plt.figure(figsize=(16, 9))
    plt.plot(r_range, vels_given, label="Given terminal velocity")
    plt.plot(r_range, vels_manual, label="Manual terminal velocity")
    plt.legend()
    plt.grid()
    plt.xlabel("Initial Droplet radius (m)", fontsize=14)
    plt.xticks(fontsize=14)
    flt.yticks(fontsize=14)
    plt.ylabel("Terminal Velocity (ms^-1)", fontsize=14)
    plt.title("Terminal velocity of droplet of given size", fontsize=20)
    plt.savefig("TerminalVelocity.png", dpi=1200)
    plt.show()