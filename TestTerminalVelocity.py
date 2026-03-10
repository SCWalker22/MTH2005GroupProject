import numpy as np
import matplotlib.pyplot as plt

Rho_w = 1000.0
pi = np.pi
g = 9.81

#Copy from here down

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

# Stop copying here
if __name__=="__main__":
    print(terminal_velocity(1e-6, 1.225))
        