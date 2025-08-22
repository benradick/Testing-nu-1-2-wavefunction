###-------------------------------------------------------------------------###
###--------------------- Electorn density no vortex ------------------------###
###-------------------------------------------------------------------------###
"""This file plots rho(theta), the electron
density of the nu=1/2 state. """
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###


from Wavefunction_CEL import Psi
import numpy as np
import matplotlib.pyplot as plt
import time
pi = np.pi
def sin(x):
    return np.sin(x)

def plot_electron_density(N, no_samples):
    theta_values = np.linspace(0.01,pi-0.01, 100) #avoid singular endpoints
    rho_values = np.zeros(len(theta_values))
    for i in range(len(theta_values)):
        if i==0:
            init = time.time()
        rho_values[i]= rho(theta_values[i],N, no_samples)
        if i == 0:
            fin = time.time()
            elapsed =( fin - init)*len(theta_values)
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f'Estimated time = {hours}h {minutes}m {seconds}s')
        print('Completed ' + str(i) + '/' + str(len(theta_values)))
    # ✅ Plot
    plt.figure(figsize=(6, 4))
    plt.plot(theta_values, rho_values, marker='o', linestyle='-', color='b', label=r'$\rho(\theta)$')
    plt.xlabel(r'$\theta$ (radians)')
    plt.ylabel(r'$\rho(\theta)$')
    plt.title(f'Electron Density for N={N}, no_samples={no_samples}\n Time to run = {hours}h {minutes}m {seconds}s')
    plt.grid(True)
    plt.legend()
    plt.ylim((0, 1.2*max(rho_values)))
    plt.show()
        
def rho(theta,N, no_samples):
    V = (2*pi**2)**(N-1)
    m_sum =0
    for m in range(no_samples):
        X_m = np.zeros((N-1,2)) 
        # X_m[i][0]= theta_{i+2}
        # X_m[i][1] = phi_{i+2}
        for i in range(N-1):
            X_m[i][0]=np.random.uniform(0,pi)
            X_m[i][1]=np.random.uniform(0, 2*pi)
        #print(X_m)
        m_sum += f(theta, X_m,N)
    return V / no_samples *m_sum

def f(theta, X_m,N):
    total=1 
    for i in range(N-1):
        theta_i_m = X_m[i][0]
        total *= sin(theta_i_m)
    Omega = np.array([theta, 0])
    positions = np.vstack([Omega, X_m])  # shape (N, 2) - need to check this does as expected
    total *= np.abs(Psi(positions, N))**2
    return total
    
    

plot_electron_density(4,10000)


