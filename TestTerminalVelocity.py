import numpy as np
import matplotlib.pyplot as plt

Rho_w = 1000.0
pi = np.pi

def down_force(r: float) -> float:
    """
    Calculate downwards force on a droplet

    Args:
        r: Radius

    Returns:
        Downwards force (N)
    """
    mass: float = Rho_w*pi*r**3
    g = 9.81
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

# start_height = 500

# dt = 0.01
# end_t = 1
# t = 0
# v = 0
# r = 1e-6
# rho = 1.225
# vels: list[float] = [v]
# times: list[float] = [t]
# while t < end_t:
#     force = down_force(r) - drag(v, r, rho)
#     mass: float = Rho_w*pi*r**3
#     accel = force/mass
#     v += dt*accel
#     vels.append(v)
#     t += dt
#     times.append(t)

# plt.plot(times, vels)
# plt.show()

def terminal_velocity(r: float, rho: float) -> float:
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

print(terminal_velocity(1e-6, 1.225))
        