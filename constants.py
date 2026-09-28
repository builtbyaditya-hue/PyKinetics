import numpy as np

# Physical constants for U-235 Thermal Fission
# These are standard textbook values used in reactor kinetics.

# Prompt neutron generation time (Lambda) in seconds
LAMBDA_PROMPT = 1e-4 

# Total delayed neutron fraction (beta)
BETA_TOTAL = 0.0065

# Delayed neutron fractions for the 6 precursor groups (beta_i)
BETA_I = np.array([0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027])

# Decay constants for the 6 precursor groups (lambda_i) in 1/seconds
LAMBDA_I = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
# --- THERMAL FEEDBACK CONSTANTS ---

# Initial core temperature in Celsius
T_INITIAL = 300.0 

# Temperature reactivity coefficient in Dollars per degree C (Negative value!)
ALPHA_T = -0.01 

# Rate at which power increases temperature (Degrees C per second per unit power)
HEATING_RATE = 50.0 

# Rate at which coolant removes heat (1/second)
COOLING_RATE = 0.5