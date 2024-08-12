# -*- coding: utf-8 -*-
"""
Fitting of the Nelson-Siegel parameters

@author: Marie Thibeau
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#%%

# Sample data: Belgian state bonds on the 28th of June 2024.
maturities = np.arange(1,31,1)
yields = np.array([0.0343, 0.0314, 0.0291, 0.0283, 0.0284, 0.0289, 0.0294, 0.0301, 
                   0.0307, 0.0313, 0.0318, 0.0324, 0.0329, 0.0333, 0.0337, 0.0341,
                   0.0345, 0.0348, 0.0351, 0.0353, 0.0356, 0.0358, 0.0359, 0.0361,
                   0.0362, 0.0363, 0.0363, 0.0363, 0.0363, 0.0362])

# Define the Nelson-Siegel model for zero-coupon yields
def nelson_siegel_yield(t, b0, b10, b11, c1):
    term1 = b0
    term2 = (b10 / (c1 * t)) * (1 - np.exp(-c1 * t))
    term3 = (b11 / (c1 ** 2 * t)) * (1 - (c1 * t + 1) * np.exp(-c1 * t))
    return term1 + term2 + term3

# Objective function to minimize: Sum of squared errors between model and observed yields
def objective(params):
    b0, b10, b11, c1 = params
    model_yields = nelson_siegel_yield(maturities, b0, b10, b11, c1)
    error = np.sum((model_yields - yields) ** 2)
    return error

# Initial guess for the parameters
initial_guess = [0.02, -0.01, 0.01, 0.1]

# Optimization
result = minimize(objective, initial_guess, method='L-BFGS-B')
b0_opt, b10_opt, b11_opt, c1_opt = result.x

print(f"Optimized Parameters:\n b0  =  {b0_opt}\n b10 =  {b10_opt}\n b11 = {b11_opt}\n c1  =  {c1_opt}")

# Plot the observed yields vs. the fitted Nelson-Siegel curve
plt.figure()
plt.scatter(maturities, yields*100, color='red', label='Observed Yields')
fitted_yields = nelson_siegel_yield(maturities, b0_opt, b10_opt, b11_opt, c1_opt)
plt.plot(maturities, fitted_yields*100, color='blue', label='Fitted Nelson-Siegel Curve')
plt.xlabel('Maturity [Years]')
plt.ylabel('Yield [%]')
plt.legend()
plt.grid(True)
plt.savefig('NS_params_opt.png')
plt.show()
