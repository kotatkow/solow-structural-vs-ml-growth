import numpy as np

def solow_simulation(alpha=0.33, s=0.2, n=0.02, g=0.02, delta=0.05,
                     k0=0.1, T=100, dt=0.1):
    """
    Simulates Solow model using Euler discretization.
    Returns time path of capital per effective worker.
    """
    
    k = np.zeros(int(T/dt))
    k[0] = k0
    
    for t in range(1, len(k)):
        dk = s * (k[t-1] ** alpha) - (n + g + delta) * k[t-1]
        k[t] = k[t-1] + dt * dk
        
    return k

def steady_state(alpha=0.33, s=0.2, n=0.02, g=0.02, delta=0.05):
    return (s / (n + g + delta)) ** (1 / (1 - alpha))