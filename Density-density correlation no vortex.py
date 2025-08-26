###-------------------------------------------------------------------------###
###----------------- Electron-electron correlation no vortex ---------------###
###-------------------------------------------------------------------------###
"""This file plots rho_2(theta), the electron-electron correlation function for the nu=1/2 state. """
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

def plot_electron_correlation(N, no_samples):
    theta_values = np.linspace(0.05,pi-0.05, 100) #avoid singular endpoints
    rho_2_values = np.zeros(len(theta_values))
    for i in range(len(theta_values)):
        if i==0:
            init = time.time()
        rho_2_values[i]= rho_2(theta_values[i],N, no_samples)
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
    plt.plot(theta_values, rho_2_values, marker='o', linestyle='-', color='b', label=r'$\rho(\theta)$')
    plt.xlabel(r'$\theta$ (radians)')
    plt.ylabel(r'$\rho(\theta)$')
    plt.title(f'Electron-electron correlation for N={N}, no_samples={no_samples}\n Time to run = {hours}h {minutes}m {seconds}s')
    plt.grid(True)
    plt.legend()
    plt.ylim((0, 1.2*max(rho_2_values)))
    plt.show()
        
def rho_2(theta,N, no_samples):
    V = (2*pi**2)**(N-2)
    m_sum =0
    for m in range(no_samples):
        X_m = np.zeros((N-2,2)) 
        # X_m[i][0]= theta_{i+3}
        # X_m[i][1] = phi_{i+3}
        for i in range(N-2):
            X_m[i][0]=np.random.uniform(0,pi) #sampling uniformly - sin(theta) in Jacobian ensures uniform sampling over the sphere.
            X_m[i][1]=np.random.uniform(0, 2*pi)
        #print(X_m)
        m_sum += f(theta, X_m,N)
    return V / no_samples *m_sum

def f(theta, X_m,N):
    total=1 
    for i in range(N-2):
        theta_i_m = X_m[i][0]
        total *= sin(theta_i_m)
    Omega = np.array([theta, 0])
    Omega_prime = np.array([0.01,0])
    positions = np.vstack([Omega, Omega_prime, X_m])  # shape (N, 2)
    total *= np.abs(Psi(positions, N))**2
    return total
    
    

plot_electron_correlation(6,10000)


