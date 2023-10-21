#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol


# In[2]:


# Parameters
# simulation dependent
S0 = 317.55            # asset price
T = 1.0                # time in years
r = 0.0721             # risk-free rate
N = 252                # number of time steps in simulation
M = 50                 # number of simulations

# Heston dependent parameters
kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.25**2        # long-term mean of variance under risk-neutral dynamics
v0 = 0.3**2            # initial variance under risk-neutral dynamics
rho = 0.7              # correlation between returns and variances under risk-neutral dynamics
sigma = 0.4            # volatility of volatility


# In[3]:


def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M):
    """
    Outputs:
    - asset prices over time 
    - variance over time
    """
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],[rho,1]])
    
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)

    Z = np.random.multivariate_normal(mu, cov, (N,M))
    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
    
    return S, v


# In[4]:


rho_p = 0.98
rho_n = -0.98

S_p,v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma,T, N, M)


# In[5]:


fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0,T,N+1)
ax1.plot(time,S_p)
ax1.set_title('Heston Model Asset Prices')
ax1.axhline(y=S0)
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')

ax2.plot(time,v_p)
ax2.set_title('Heston Model Variance Process')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')

plt.show()

