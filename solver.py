import numpy as np
from scipy.integrate import solve_ivp
from constants import (LAMBDA_PROMPT, BETA_TOTAL, BETA_I, LAMBDA_I, 
                       ALPHA_T, HEATING_RATE, COOLING_RATE, T_INITIAL)

def point_kinetics_equations(t, y, rho_ext_func):
    """
    Calculates the derivatives for the Point Kinetics Equations with Thermal Feedback.
    y[0]   : Neutron density / Power (n)
    y[1:7] : Precursor concentrations (C_i)
    y[7]   : Core Temperature (T)
    """
    n = y[0]
    C = y[1:7]
    T = y[7]

    # 1. Calculate Reactivity ($)
    rho_rods = rho_ext_func(t)
    rho_feedback = ALPHA_T * (T - T_INITIAL)
    rho_total = rho_rods + rho_feedback

    # Convert reactivity from dollars to absolute reactivity
    rho_abs = rho_total * BETA_TOTAL

    # 2. Neutron density derivative (dn/dt)
    dn_dt = ((rho_abs - BETA_TOTAL) / LAMBDA_PROMPT) * n + np.sum(LAMBDA_I * C)

    # 3. Precursor derivatives (dC/dt)
    dC_dt = (BETA_I / LAMBDA_PROMPT) * n - LAMBDA_I * C

    # 4. Temperature derivative (dT/dt)
    # Heating from excess power minus cooling from coolant flow
    dT_dt = HEATING_RATE * (n - 1.0) - COOLING_RATE * (T - T_INITIAL)

    return [dn_dt] + dC_dt.tolist() + [dT_dt]


def solve_reactor_response(rho_ext_func, t_end=30.0):
    """
    Runs the simulation from t=0 to t_end.
    """
    t_span = (0, t_end)
    
    # Set perfectly stable initial conditions
    n0 = 1.0
    C0 = (BETA_I / (LAMBDA_PROMPT * LAMBDA_I)) * n0
    T0 = T_INITIAL
    
    # y0 is our starting state vector (8 items long)
    y0 = [n0] + C0.tolist() + [T0]
    
    # Run the rigorous SciPy solver
    solution = solve_ivp(
        point_kinetics_equations,
        t_span,
        y0,
        args=(rho_ext_func,),
        method='BDF',
        dense_output=True
    )
    return solution