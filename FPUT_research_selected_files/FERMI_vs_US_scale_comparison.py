"""
Created on Thu Mar  6 10:45:17 2025

@author: matteolotriglia
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------
# DEFINITION OF COMMON FUNCTIONS
# ----------------------------------------

# Force
def F(q, k, m, alpha, beta, N):
    F = np.zeros(N)
    for j in range(1, N - 1):
        F[j] = k / m * (q[j + 1] - 2 * q[j] + q[j - 1]) + alpha / m * (
                (q[j + 1] - q[j]) ** 2 - (q[j] - q[j - 1]) ** 2) + beta / m * (
                       (q[j + 1] - q[j]) ** 3 - (q[j] - q[j - 1]) ** 3)
    return F

# Runge-Kutta Method
def RK(F, q, v, n, dt, m, alpha, beta, N, K):
    for i in tqdm(range(n), desc="Evolution with Runge-Kutta"):
        k1_q = v[:, i]
        k1_v = F(q[:, i], K, m, alpha, beta, N)
        k2_q = v[:, i] + k1_v * dt / 2
        k2_v = F(q[:, i] + k1_q * dt / 2, K, m, alpha, beta, N)
        k3_q = v[:, i] + k2_v * dt / 2
        k3_v = F(q[:, i] + k2_q * dt / 2, K, m, alpha, beta, N)
        k4_q = v[:, i] + k3_v * dt
        k4_v = F(q[:, i] + k3_q * dt, K, m, alpha, beta, N)

        q[:, i + 1] = q[:, i] + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) * dt / 6
        v[:, i + 1] = v[:, i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6

        # Boundary conditions for Fermi method
        if N == 33:
            q[0, i + 1] = 0
            q[N-1, i + 1] = 0
            v[0, i + 1] = 0
            v[N-1, i + 1] = 0

    return q, v

# Omega calculation for Fermi
def omega_fermi(N):
    omega = []
    for mod in range(0, N):
        omega.append(np.sqrt(2) * np.abs(np.sin((mod+1) * np.pi / (2*N))))
    return np.array(omega)

# Omega calculation for our method
def w_fun(k, K, m, N):
    omega = 2 * np.sqrt(K / m) * np.abs(np.sin(np.pi * (k) / (N-1)))
    return omega

# Transformation from q to a for Fermi
def from_q_to_a_fermi(q, N, n_iter):
    a = np.zeros((N-1, n_iter + 1))
    for timestep in tqdm(range(n_iter + 1), desc="Transforming to a (Fermi)"):
        for mode_a in range(N-1):
            for mode_q in range(1, N-1):
                a[mode_a, timestep] += np.sqrt(2.0 / (N - 1))*q[mode_q, timestep] * np.sin((mode_q) * (mode_a+1) * np.pi / (N-1))
    return a

# Transformation from q to a complex for our method
def from_q_to_a_complex(q, v, N, w, n_iter):
    a = np.zeros((N-1, n_iter + 1), dtype=complex)
    m = 1  # Mass is 1 in both scripts
    
    for timestep in tqdm(range(n_iter + 1), desc="Transforming to a (Complex)"):
        # Using Fourier transform
        Q = np.fft.fft(q[:-1, timestep])/np.sqrt(N-1)
        P = np.fft.fft(v[:-1, timestep])/np.sqrt(N-1)
        
        # Calculating complex amplitudes
        a[1:, timestep] = (1 / np.sqrt(2 * m * w[1:])) * (m * w[1:] * Q[1:] + 1j * P[1:])
    
    return a

# Energy calculation for Fermi method
def mode_energy_fermi(a, a_dot, omega_f, mode):
    T = []
    V = []
    E = []
    for j in range(0, len(a.T) - 1):
        T.append(0.5 * a_dot[mode, j + 1] ** 2)
        V.append(omega_f[mode] ** 2 * a[mode, j + 1] ** 2)
        E.append(T[j] + V[j])
    return np.array(E), np.array(T), np.array(V)

# Energy calculation for our method
def mode_energy_a_complex(a, omega_f, mode):
    V = []
    E = []
    for j in range(0, len(a.T) - 1):
        V.append(omega_f[mode] * np.abs(a[mode, j + 1]) ** 2)
        E.append(V[j])
    return np.array(E), np.array([]), np.array(V)

# Setting initial conditions
def set_init_cond(q, v, N, string_cond):
    if string_cond == "sine":
        # Sine wave for Fermi
        L = 10
        d = np.linspace(0, L, N)
        q[:, 0] = np.sin(np.pi * d / L)
    elif string_cond == "sine_periodic":
        # Periodic sine wave for our method
        L = 10
        d = np.linspace(0, L, N)
        q[:, 0] = np.sin(2 * np.pi * d / L)
    
    return q, v

# ----------------------------------------
# MAIN PARAMETERS
# ----------------------------------------
pi = np.pi
m = 1       # Mass
K = 1       # Elastic constant
alpha = 0.25  # Quadratic coefficient
beta = 0    # Cubic coefficient

# Time parameters
T = 9000      # Final time
dt = 0.2      # Time step
n_iter = int(T / dt)  # Number of iterations
t = np.linspace(0, T, n_iter + 1)

# ----------------------------------------
# FERMI METHOD (N_plus_1 = 33)
# ----------------------------------------
print("\nSTARTING SIMULATION WITH FERMI'S METHOD (N=33)")
N_fermi_plus_1 = 33
k_fermi = np.arange(0, N_fermi_plus_1, 1)
w_fermi = omega_fermi(N_fermi_plus_1-1)

# Initial conditions
q_fermi = np.zeros((N_fermi_plus_1, n_iter + 1))
v_fermi = np.zeros((N_fermi_plus_1, n_iter + 1))
q_fermi, v_fermi = set_init_cond(q_fermi, v_fermi, N_fermi_plus_1, "sine")

# Evolution
print("\nSystem evolution with Fermi:")
start = time.time()
q_RK_fermi, v_RK_fermi = RK(F, q_fermi, v_fermi, n_iter, dt, m, alpha, beta, N_fermi_plus_1, K)
end = time.time()
print(f'Elapsed time with Fermi: {end - start:.2f} seconds')

# Transformation to a, a_dot variables
print("\nSwitch to a (Fermi):")
a_fermi = from_q_to_a_fermi(q_RK_fermi, N_fermi_plus_1, n_iter)
a_dot_fermi = from_q_to_a_fermi(v_RK_fermi, N_fermi_plus_1, n_iter)

# ----------------------------------------
# OUR METHOD (N_plus_1 = 65)
# ----------------------------------------
print("\nSTARTING SIMULATION WITH OUR METHOD (N=65)")
N_custom_plus_1 = 65
k_custom = np.arange(0, N_custom_plus_1-1, 1)
w_custom = w_fun(k_custom, K, m, N_custom_plus_1)

# Initial conditions
q_custom = np.zeros((N_custom_plus_1, n_iter + 1))
v_custom = np.zeros((N_custom_plus_1, n_iter + 1))
q_custom, v_custom = set_init_cond(q_custom, v_custom, N_custom_plus_1, "sine_periodic")

# Evolution
print("\nSYSTEM EVOLUTION WITH OUR METHOD:")
start = time.time()
q_RK_custom, v_RK_custom = RK(F, q_custom, v_custom, n_iter, dt, m, alpha, beta, N_custom_plus_1, K)
end = time.time()
print(f'Elapsed time with our method: {end - start:.2f} seconds')

# Transformation to a variables
print("\nSwitch to a (ours):")
a_custom = from_q_to_a_complex(q_RK_custom, v_RK_custom, N_custom_plus_1, w_custom, n_iter)

# ----------------------------------------
# ENERGY CALCULATION AND RATIO
# ----------------------------------------
print("\nCalculating energies and ratio:")

# Number of modes to analyze
num_modes = 5

# Arrays for energies
energy_fermi = []
energy_custom = []
ratio_energy = []

# Energy calculation for first 5 modes
for mode in tqdm(range(num_modes), desc="Calculating energy for each mode"):
    # Energy for Fermi method
    E_fermi, _, _ = mode_energy_fermi(a_fermi, a_dot_fermi, w_fermi, mode)
    energy_fermi.append(E_fermi)
    
    # Energy for custom method (we must use mode+1 because mode 0 is omitted)
    E_custom, _, _ = mode_energy_a_complex(a_custom, w_custom, mode+1)
    energy_custom.append(E_custom)
    
    # Calculating ratio
    # Ensure they have the same length
    min_len = min(len(E_fermi), len(E_custom))
    ratio = E_fermi[:min_len] / (E_custom[:min_len] + 1e-10)  # Avoid division by zero
    ratio_energy.append(ratio)

# ----------------------------------------
# RESULTS VISUALIZATION
# ----------------------------------------
print("\nPlotting results:")

# Plot energies for each mode
plt.figure(figsize=(12, 8))
for i in range(num_modes):
    plt.plot(t[:-1], energy_fermi[i], label=f"Fermi mode {i+1}")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy of the first 5 modes with Fermi')
plt.grid(True)
plt.legend()
#plt.savefig('energie_fermi.png')
plt.show()

plt.figure(figsize=(12, 8))
for i in range(num_modes):
    plt.plot(t[:-1], energy_custom[i], label=f"Our mode {i+1}")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy of the first 5 modes with our method')
plt.grid(True)
plt.legend()
#plt.savefig('energie_custom.png')
plt.show()

# Plot energy ratio
plt.figure(figsize=(15, 10))
for i in range(num_modes):
    plt.subplot(num_modes, 1, i+1)
    plt.plot(t[:-1], ratio_energy[i])
    plt.ylabel(f'Ratio E_Fermi/E_ours\nMode {i+1}')
    plt.grid(True)
    if i == num_modes-1:
        plt.xlabel('Time')
plt.suptitle('Mode Energy Ratio Fermi/Ours')
plt.tight_layout()
#plt.savefig('rapporto_energie.png')
plt.show()

# Logarithmic plot of the ratio
plt.figure(figsize=(12, 8))
for i in range(num_modes):
    plt.semilogy(t[:-1], ratio_energy[i], label=f"Mode {i+1}")
plt.xlabel('Time')
plt.ylabel('Energy Ratio (log scale)')
plt.title('Mode Energy Ratio Fermi/Ours (log scale)')
plt.grid(True)
plt.legend()
#plt.savefig('rapporto_energie_log.png')
plt.show()

print("\nCOMPLETE.")