"""
D. G. Partridge, ESE, University of Exeter

Program to solve growth of monodisperse cloud droplet population
in an ascending air parcel
"""

import numpy as np

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

# Saturation vapour pressure will be provided by your svp() function
# from svp import svp   # Uncomment if svp is in a separate file

# --------------------------------------------------------------------------

# %%Advanced: In reality the Thermal Conductivty & Diffusivity are not constant, but depend on the temperature and pressure.
# They also need to include kinetic effects (i.e., the effects of condensation and accommodation coefficients; e.g., Fukuta and Walter 1970).
# Assuming the constant values above is adequate for your project. If you would like to account for the temperature dependance
# please do, you can find further details in the course textbook or in "Lohmann, an introduction to clouds", eq. 7.27, page 192.

# --------------------------------------------------------------------------
# Define initial vertical velocity and droplet number concentration

# Example placeholders — replace with your values
W = None                      # Vertical velocity (m/s)
NDropletDensity = None        # Droplet number concentration (#/m^3)

# --------------------------------------------------------------------------
# Main program starts here

def main():
    """
    Main model driver.
    Implement parcel ascent and droplet growth equations here.
    """
    pass


if __name__ == "__main__":
    main()

