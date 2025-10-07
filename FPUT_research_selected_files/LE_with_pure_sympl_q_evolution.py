
import numpy as np
from scipy.fftpack import fft, ifft
import time
import warnings
import Delta_kronecker as d
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Style
import os


warnings.filterwarnings('ignore')


###############################
#                             #
#      DEFINING FUNCTIONS     #
#                             #
###############################

def from_our_Q_to_a_complex_single_step(Q, P, N, w):
    """
    Transforms the real-space Fourier coefficients (Q, P) into complex
    normal mode amplitudes 'a'.

    Args:
        Q (np.ndarray): Fourier coefficients of the positions.
        P (np.ndarray): Fourier coefficients of the momenta.
        N (int): Number of oscillators.
        w (np.ndarray): Array of linear frequencies.

    Returns:
        np.ndarray: Array of complex normal mode amplitudes 'a'.
    """
    a_rec = np.zeros(N) * 1j
    
    a_rec[1:] = (1 / np.sqrt(2 * m * w[1:])) * (m * w[1:] * Q[1:] + 1j * P[1:])
    a_rec[0] = 0
            
    return a_rec

def from_a_complex_to_Q(a0):
    """
    Transforms the complex normal mode amplitudes 'a' back into the
    Fourier coefficients of position (Q) and momentum (P). This is the
    inverse transformation of 'from_our_Q_to_a_complex_single_step'.

    Args:
        a0 (np.ndarray): Array of complex normal mode amplitudes 'a'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the Fourier
        coefficients for position (Q_old) and momentum (P_old).
    """
    a0_cc = np.conj(a0)
    a0_cc[1:] = a0_cc[:0:-1]
    
    Q_k0 = np.zeros_like(a0, dtype=complex)
    P_k0 = np.zeros_like(a0, dtype=complex)
    
    Q_k0[1:] = 1 / np.sqrt(2 * m * w[1:]) * (a0[1:] + a0_cc[1:])
    Q_k0[0] = 0
    
    P_k0[1:] = -1j * np.sqrt(m * w[1:] / 2) * (a0[1:] - a0_cc[1:])
    P_k0[0] = 0
    
    Q_old = Q_k0
    P_old = P_k0

    return Q_old, P_old

def from_a_complex_to_q(a0):
    """
    Transforms the complex normal mode amplitudes 'a' into real-space
    positions (q) and momenta (p) by first converting to Fourier
    coefficients (Q, P) and then applying an inverse FFT.

    Args:
        a0 (np.ndarray): Array of complex normal mode amplitudes 'a'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the real-space
        positions (q_old) and momenta (p_old).
    """
    a0_cc = np.conj(a0)
    a0_cc[1:] = a0_cc[:0:-1]
    
    # Mapping back to Fourier amplitudes and "momentum"
    Q_k0 = np.zeros_like(a0, dtype=complex)
    P_k0 = np.zeros_like(a0, dtype=complex)

    Q_k0[1:] = 1 / np.sqrt(2 * m * w[1:]) * (a0[1:] + a0_cc[1:])
    Q_k0[0] = 0
    
    P_k0[1:] = -1j * np.sqrt(m * w[1:] / 2) * (a0[1:] - a0_cc[1:])
    P_k0[0] = 0
    
    # Discrete IFFT to get the real variables
    q_0 = N * ifft(Q_k0)
    p_0 = N * ifft(P_k0)

    # Storing the real part for time evolution
    q_old = q_0.real
    p_old = p_0.real

    return q_old, p_old

def w_fun(k):
    """
    Calculates the dispersion relation for the FPUT system.

    Args:
        k (np.ndarray or float): An array of wavenumbers or a single wavenumber.

    Returns:
        np.ndarray or float: The corresponding angular frequencies (omega)
        for the linear approximation of the system.
    """
    global K, m, N
    
    omega = np.abs(2 * np.sqrt(K/m) * np.sin(pi * k / N))
    
    return omega

def RHS_super_roll_fdo(q):
    """
    Computes the right-hand-side of the FPUT equation of motion (i.e., the force)
    using an optimized method with np.roll for periodic boundary conditions.
    'fdo' stands for 'forward difference only'.

    Args:
        q (np.ndarray): The array of particle positions at a given time.

    Returns:
        np.ndarray: The forces acting on each particle (p_dot).
    """
    global K, alpha, beta
          
    q_diff_forward = np.roll(q, -1) - q
    q_diff_forward_sq = q_diff_forward**2
    q_diff_forward_cube = q_diff_forward_sq * q_diff_forward
    rhs_terms = K * q_diff_forward + alpha * q_diff_forward_sq + beta * q_diff_forward_cube

    return rhs_terms - np.roll(rhs_terms, 1)

def symplectic(q_old, p_old, RHS_q, omega, kappa):
    """
    Evolves the system for one time step using a Yoshida 8th-order
    symplectic integrator. It updates positions (q) and momenta (p).

    Args:
        q_old (np.ndarray): Positions at the beginning of the time step.
        p_old (np.ndarray): Momenta at the beginning of the time step.
        RHS_q (function): The function that calculates the forces (e.g., RHS_super_roll_fdo).
        omega (np.ndarray): Not used in this implementation but kept for signature compatibility.
        kappa (float): Not used in this implementation but kept for signature compatibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: The updated positions (q_new) and momenta (p_new).
    """
    k = 2 * ms + 2
    for r in range(0, k):
        q_new = q_old + dt * c_sympl[r] * p_old
        p_new = p_old + dt * d_sympl[r] * RHS_q(q_new)

        q_old = q_new
        p_old = p_new

    return q_new, p_new

def RHS_a(a):
    """
    Calculates the right-hand-side of the equation of motion for the complex
    amplitudes 'a', including linear, quadratic (triad), and cubic (quartet)
    interaction terms. This is used for direct evolution in 'a' space.

    Args:
        a (np.ndarray): Array of complex amplitudes 'a' at a given time.

    Returns:
        np.ndarray: The time derivative of 'a' (da/dt).
    """
    global w, k 
    global V1,V2,V3,T1,T2,T3,T4
    global delta3_1, delta3_2, delta3_3, delta4_1, delta4_2 ,delta4_3 ,delta4_4
    
    # Linear contribution
    a_lin = w * a
    
    a = a.reshape(N,1) # Reshape array into a column vector for matrix multiplication
    a_t = a.T # Transpose of a to get a row vector
    
    # Get the complex conjugate
    a_cc = np.conj(a)
    a_cc_t = a_cc.T
    
    a_d1 = a.reshape(N,1,1)
    a_d2 = a.reshape(1,N,1)
    a_d3 = a.reshape(1,1,N)
    
    a_d1_cc = a_cc.reshape(N,1,1)
    a_d2_cc = a_cc.reshape(1,N,1)
    a_d3_cc = a_cc.reshape(1,1,N)
    
    # Triad contribution (quadratic in a)     
    a2 = (( V1*(a_t*a)*delta3_1 \
            - 2*V2*(a_cc_t*a).T*delta3_2 \
            - V3*(a_cc_t*a_cc)*delta3_3).sum(axis=2)).sum(axis=1)
    
    # Fourth-wave resonance (cubic contribution in a)
    a3 = (((T1*a_d1*a_d2*a_d3*delta4_1 \
            + 3*T2*a_d1_cc*a_d2*a_d3*delta4_2 \
            + 3*T3*a_d1_cc*a_d2_cc*a_d3*delta4_3 \
            + T4*a_d1_cc*a_d2_cc*a_d3_cc*delta4_4 \
            ).sum(axis=3)).sum(axis=2)).sum(axis=1)

    da_dt = a_lin + a2 + a3
    
    return da_dt.reshape(N)

def RK(y_old):
    """
    Evolves the complex amplitudes 'a' for one time step using a specific
    4th-order Runge-Kutta method, applied in two half-steps.

    Args:
        y_old (np.ndarray): The complex amplitudes 'a' at the start of the step.

    Returns:
        np.ndarray: The updated complex amplitudes 'a' after one time step.
    """
    k1 = (-1j) * RHS_a(y_old) * (dt/2)
    k2 = (-1j) * RHS_a(y_old + k1/2) * (dt/2)
    k3 = (-1j) * RHS_a(y_old + k1/4 + k2/4) * (dt/2)
    k4 = (-1j) * RHS_a(y_old - k2 + 2*k3) * (dt/2)
    
    y_t_half = y_old + (k1 + 4*k3 + k4)/6

    k5 = (-1j) * RHS_a(y_t_half) * (dt/2)
    k6 = (-1j) * RHS_a(y_t_half + k5/2) * (dt/2)
    k7 = (-1j) * RHS_a(y_t_half + k5/4 + k6/4) * (dt/2)
    k8 = (-1j) * RHS_a(y_t_half - k6 + 2*k7) * (dt/2)
        
    y = y_t_half + (k5 + 4*k7 + k8)/6
    
    return y

def d_to_a(d):
    """
    Performs a near-identity transformation from variables 'd' to 'a'.
    This transformation is used in perturbation theory to simplify the
    Hamiltonian by removing non-resonant terms up to a certain order.
    This version includes quadratic, cubic, and quartic corrections.

    Args:
        d (np.ndarray): The array of complex variables 'd'.

    Returns:
        np.ndarray: The transformed array of complex variables 'a'.
    """
    global A1, A2, A3, B1, B2, B3, B4, C1, C2, C3, C4, C5
    global delta3_1, delta3_2, delta3_3, delta4_1, delta4_2, delta4_3, delta4_4, delta5_1, delta5_2, delta5_3, delta5_4, delta5_5
    
    # Implementing the second near-identity transformation to get b from c
    b = d.reshape(1, N)
    b_t = b.T
    b_cc = np.conj(b)
    b_cc_t = b_cc.T

    # Triad contribution
    b2 = ((A1 * (b * b_t) * delta3_1 \
           + A2 * (b * b_cc_t) * delta3_2 \
           + A3 * (b_cc * b_cc_t) * delta3_3).sum(axis=2)).sum(axis=1)

    c = d.reshape(N, 1)
    c_cc = np.conj(c)

    c_d1 = c.reshape(N, 1, 1)
    c_d2 = c.reshape(1, N, 1)
    c_d3 = c.reshape(1, 1, N)
    c_d1_cc = c_cc.reshape(N, 1, 1)
    c_d2_cc = c_cc.reshape(1, N, 1)
    c_d3_cc = c_cc.reshape(1, 1, N)
    
    # Fourth-wave resonance (cubic contribution in a)
    c3 = (((B1 * c_d1 * c_d2 * c_d3 * delta4_1 + \
            B2 * c_d1_cc * c_d2 * c_d3 * delta4_2 + \
            B3 * c_d1_cc * c_d2_cc * c_d3 * delta4_3 + \
            B4 * c_d1_cc * c_d2_cc * c_d3_cc * delta4_4).sum(axis=3)).sum(axis=2)).sum(axis=1)

    d_reshaped = d.reshape(N, 1)
    d_cc = np.conj(d_reshaped)

    d_d1 = d_reshaped.reshape(N, 1, 1, 1)
    d_d2 = d_reshaped.reshape(1, N, 1, 1)
    d_d3 = d_reshaped.reshape(1, 1, N, 1)
    d_d4 = d_reshaped.reshape(1, 1, 1, N)
    d_d1_cc = d_cc.reshape(N, 1, 1, 1)
    d_d2_cc = d_cc.reshape(1, N, 1, 1)
    d_d3_cc = d_cc.reshape(1, 1, N, 1)
    d_d4_cc = d_cc.reshape(1, 1, 1, N)

    # Five-wave resonance (quartic contribution in a)
    d4 = ((((C1 * d_d1 * d_d2 * d_d3 * d_d4 * delta5_1 + \
             C2 * d_d1_cc * d_d2 * d_d3 * d_d4 * delta5_2 + \
             C3 * d_d1_cc * d_d2_cc * d_d3 * d_d4 * delta5_3 + \
             C4 * d_d1_cc * d_d2_cc * d_d3_cc * d_d4 * delta5_4 + \
             C5 * d_d1_cc * d_d2_cc * d_d3_cc * d_d4_cc * delta5_5).sum(axis=4)).sum(axis=3)).sum(axis=2)).sum(axis=1)

    a = d.reshape(N) + b2 + c3 + d4
    return a

def d_to_c(d):
    """
    Performs a near-identity transformation from variables 'd' to 'c'.
    This transformation adds the fifth-order (quartic) correction terms.
    NOTE: This transformation is not exact by itself but part of a larger transformation.

    Args:
        d (np.ndarray): The array of complex variables 'd'.

    Returns:
        np.ndarray: The transformed array of complex variables 'c'.
    """
    global C1, C2, C3, C4, C5
    global delta5_1, delta5_2, delta5_3, delta5_4, delta5_5
    
    d_reshaped = d.reshape(N, 1)
    d_cc = np.conj(d_reshaped)

    d_d1 = d_reshaped.reshape(N, 1, 1, 1)
    d_d2 = d_reshaped.reshape(1, N, 1, 1)
    d_d3 = d_reshaped.reshape(1, 1, N, 1)
    d_d4 = d_reshaped.reshape(1, 1, 1, N)
    d_d1_cc = d_cc.reshape(N, 1, 1, 1)
    d_d2_cc = d_cc.reshape(1, N, 1, 1)
    d_d3_cc = d_cc.reshape(1, 1, N, 1)
    d_d4_cc = d_cc.reshape(1, 1, 1, N)
    
    # Five-wave resonance (quartic contribution in a)
    d4 = ((((C1 * d_d1 * d_d2 * d_d3 * d_d4 * delta5_1 + \
             C2 * d_d1_cc * d_d2 * d_d3 * d_d4 * delta5_2 + \
             C3 * d_d1_cc * d_d2_cc * d_d3 * d_d4 * delta5_3 + \
             C4 * d_d1_cc * d_d2_cc * d_d3_cc * d_d4 * delta5_4 + \
             C5 * d_d1_cc * d_d2_cc * d_d3_cc * d_d4_cc * delta5_5).sum(axis=4)).sum(axis=3)).sum(axis=2)).sum(axis=1)

    c = d.reshape(N) + d4
    return c

def c_to_a(c):
    """
    Performs a near-identity transformation from variables 'c' to 'a'.
    This transformation adds the third-order (quadratic) and fourth-order (cubic)
    correction terms. It is part of the full transformation from 'd' to 'a'.

    Args:
        c (np.ndarray): The array of complex variables 'c'.

    Returns:
        np.ndarray: The transformed array of complex variables 'a'.
    """
    global A1, A2, A3, B1, B2, B3, B4
    global delta3_1, delta3_2, delta3_3, delta4_1, delta4_2, delta4_3, delta4_4
    
    # Using 'b' as a temporary variable name consistent with theory
    b = c.reshape(1, N)
    b_t = b.T
    b_cc = np.conj(b)
    b_cc_t = b_cc.T

    # Triad contribution
    b2 = ((A1 * (b * b_t) * delta3_1 \
           + A2 * (b * b_cc_t) * delta3_2 \
           + A3 * (b_cc * b_cc_t) * delta3_3).sum(axis=2)).sum(axis=1)

    c_reshaped = c.reshape(N, 1)
    c_cc = np.conj(c_reshaped)

    c_d1 = c_reshaped.reshape(N, 1, 1)
    c_d2 = c_reshaped.reshape(1, N, 1)
    c_d3 = c_reshaped.reshape(1, 1, N)
    c_d1_cc = c_cc.reshape(N, 1, 1)
    c_d2_cc = c_cc.reshape(1, N, 1)
    c_d3_cc = c_cc.reshape(1, 1, N)
    
    # Fourth-wave resonance (cubic contribution in a)
    c3 = (((B1 * c_d1 * c_d2 * c_d3 * delta4_1 + \
            B2 * c_d1_cc * c_d2 * c_d3 * delta4_2 + \
            B3 * c_d1_cc * c_d2_cc * c_d3 * delta4_3 + \
            B4 * c_d1_cc * c_d2_cc * c_d3_cc * delta4_4).sum(axis=3)).sum(axis=2)).sum(axis=1)

    a = c.reshape(N) + b2 + c3
    return a

def set_antisymmetric_IC(N, amplitude):
    """
    Sets antisymmetric initial conditions for the particle positions.
    The first and last oscillators are fixed at 0. The second oscillator
    has the maximum amplitude, and the amplitudes decrease linearly to the
    center, with the other half being antisymmetric.

    Args:
        N (int): Number of oscillators.
        amplitude (float or complex): Amplitude of the second oscillator.
        
    Returns:
        list: A list of all initial amplitudes.
    """
    amplitudes = [0] * N  # Initialize all to zero
    
    # Set the value of the second oscillator (index 1)
    amplitudes[1] = amplitude
    
    # Calculate the central point (considering N-2 internal oscillators)
    center = (N - 1) / 2
    
    # Calculate the decrement step
    step = amplitude / (center - 1)
    
    # Set the values of the internal oscillators (from 1 to N-2)
    for i in range(1, N - 1):
        if i < center:
            amplitudes[i] = amplitude - (i - 1) * step
        elif i > center:
            amplitudes[i] = -amplitudes[N - 1 - i]
    
    return amplitudes


###############################
#                             #
#             MAIN            #
#                             #
###############################

start_time = time.time()
eps = np.finfo(float).eps
pi = np.pi


'''
SWITCHES DEFINITIONS
'''
is_q_evolution = True         # Choose between a and q evolution
has_antisymmetric_IC = False  # Substitute usual IC with antisymmetric ones
is_zero_mode_fixed = True     # Toggle ON for FPUT case

is_analyzing_recurrence = True
is_broken = False
'''
END SWITCHES DEFINITIONS
'''
print('')
print(Fore.CYAN + "RUNNING THE CODE WITH THE FOLLOWING CONDITIONS:" + Style.RESET_ALL)
print("is_q_evolution: ", is_q_evolution)
print("has_antisymmetric_IC: ", has_antisymmetric_IC)
print("is_zero_mode_fixed: ", is_zero_mode_fixed)
print('')

if is_q_evolution: # Manually set main parameters with q evolution
    N = 8 
    alpha = 0.25
    beta = 0

else:
    # Load calculated tensors from specific subdirecttory 
    import OPT_recover_tensors

    tensors = OPT_recover_tensors.load_all_tensors(9, 1, 0.05)

    # Unpack tensors
    for name, tensor in tensors.items():
        # Create a variable with the same name as the tensor
        globals()[name] = tensor 
    
    N = OPT_recover_tensors.N_stored
    alpha = OPT_recover_tensors.alpha_stored
    beta = OPT_recover_tensors.beta_stored

# Parameters
L = 2*pi   # domain length 
dx = L/N   # step size
x = np.arange(0, L, dx) # 1D space grid

# Constants
K = 1 # linear coefficient
m = 1 # mass of the particles
delta = 0.4 # Perturbation

# Time conditions
dt = 0.1
T_fin = 300000 * dt
num_iter = int(T_fin/dt)

t_vec=np.arange(0,T_fin,dt)

# Parameters for symplectic method
w3 = 0.784513610477560
w2 = 0.235573213359357
w1 = -1.17767998417887
w0 = 1.315186320683906

c_sympl = [0.5 * w3, 0.5 * (w3 + w2), 0.5 * (w2 + w1), 0.5 * (w1 + w0), 0.5 * (w1 + w0), 0.5 * (w2 + w1), 0.5 * (w3 + w2),
           0.5 * w3]
d_sympl = [w3, w2, w1, w0, w1, w2, w3, 0]

ms = 3

# Initial conditions for q and p
q_0 = np.zeros(N) 
p_0 = np.zeros(N)
a0 = np.zeros(N) * 1j
c0 = np.zeros(N) * 1j
d0 = np.zeros(N) * 1j

k = np.arange(0,N,1)
w = w_fun(k)

# Storing the Kronecker deltas
if not is_q_evolution:
    n = 3
    delta3_1 = d.delta_k(k, n, 1)
    delta3_2 = d.delta_k(k, n, 2)
    delta3_3 = d.delta_k(k, n, 3)
    
    n = 4
    delta4_1 = d.delta_k(k, n, 1)
    delta4_2 = d.delta_k(k, n, 2)
    delta4_3 = d.delta_k(k, n, 3)
    delta4_4 = d.delta_k(k, n, 4)
    
    n = 5
    delta5_1 = d.delta_k(k, n, 1)
    delta5_2 = d.delta_k(k, n, 2)
    delta5_3 = d.delta_k(k, n, 3)
    delta5_4 = d.delta_k(k, n, 4)
    delta5_5 = d.delta_k(k, n, 5)

# Normal mode initial condition
if not has_antisymmetric_IC:
    
    list_in_mod = np.arange(1, N)
    
    d0[list_in_mod] = delta / (np.sqrt(w[list_in_mod]) * len(list_in_mod) * (1 - 1 / N + np.array(list_in_mod) / N))
    
    if is_q_evolution:
        a0 = d0
    else:
        c0 = d_to_c(d0)
        
        a0 = d_to_a(d0)
        ac0 = c_to_a(c0)
    
else:
    amplitude = 3*1j
    a0 = set_antisymmetric_IC(N, amplitude)
    print('')
    print("Antisymmetric IC:", a0)
    print('')
    
    if not is_q_evolution:
        a0 = np.array(a0)

# Transform IC for a into q and p
q0, p0 = from_a_complex_to_q(a0)
print('')
print("q0: ", q0)

# DEBUG: plotting ICs
plt.plot(np.real(a0), label='Re(a0)')
plt.plot(np.abs(a0), label='|a0|')
plt.plot(q0, label='q0')
plt.grid(True)
plt.title("Initial Conditions")
plt.legend(loc='best',fontsize= 'x-large')
plt.show()

a0_cc = np.conj(a0)
a0_cc[1:] = a0_cc[:0:-1]

a_old = a0 

# Perturbations
perturbation_of_ic = 10**(-5)
zero_matrix = np.zeros((N,N))
basis_real = np.identity(N)
basis_real = np.concatenate((basis_real, zero_matrix), axis=1)
basis_complx = np.identity(N)*1j
basis_complx = np.concatenate((zero_matrix, basis_complx), axis=1)
basis = np.concatenate((basis_real, basis_complx))
perturbation_basis = perturbation_of_ic*basis

# Number of Lyapunov Exponents to be considered
spectrum_num = 2*N

# Set Clones
CLONEs_a0 = []
CLONEs_a0_cc = []

for c in range(spectrum_num):
    if c < N:
        a0_perturbed = np.array(a0 + perturbation_basis[c, :N])
        a0_cc_perturbed = np.conj(a0_perturbed)
    else:
        a0_perturbed = np.array(a0 + perturbation_basis[c, N:])
        a0_cc_perturbed = np.conj(a0_perturbed)

    CLONEs_a0.append(a0_perturbed)
    CLONEs_a0_cc.append(a0_cc_perturbed)

# Initialize clones
CLONEs_old_a = CLONEs_a0
CLONEs_old_a_cc = CLONEs_a0_cc

# List to append parameters
l_list = [[] for _ in range(spectrum_num)]
Lyap_spectrum = [[] for _ in range(spectrum_num)]


# Pre-cycle parameters
T_cycle = 100 

cycle = 1
iteration = 0

########################## APPLYING CLONE METHOD ##############################

# DEBUG
if is_analyzing_recurrence:
    q_trajectory = []
    p_trajectory = []

print('')
print(Fore.CYAN + 'EVOLVING STANDARD AND CLONE TRAJECTORIES:' + Style.RESET_ALL)
print('')

for tt in tqdm(range(num_iter)):
    
    # Numerical integration method for a
    
    # TOGGLE DIRECT a vs. a(q)
    if not is_q_evolution:
        a_new = RK(a_old)
        a_new_cc = np.conj(a_new)
        
        # Break cycle if a_new is nan
        if np.isnan(a_new[1]):
            print('')
            print(f"Breaking evolution cycle since a_new is NaN at iteration {tt}...")
            print('')
            
            is_broken = True
            break
        
    else:
        Q_old, P_old = from_a_complex_to_Q(a_old)
        
        if is_analyzing_recurrence:
            # Saving trajectories:
            if is_q_evolution:
                q_current = ifft(Q_old)*N
                p_current = ifft(P_old)*N
            else:
                q_current, p_current = from_a_complex_to_q(a_old)
        
            q_trajectory.append(q_current.real)
            p_trajectory.append(p_current.real)
        
        q_new, p_new = symplectic(ifft(Q_old)*N, ifft(P_old)*N, RHS_super_roll_fdo, w, K)
        
        # Break cycle if q_new is nan
        if np.isnan(q_new[1]):
            print('')
            print(f"Breaking evolution cycle since q_new is NaN at iteration {tt}...")
            print('')
            
            is_broken = True
            break
        
        aq_new = from_our_Q_to_a_complex_single_step(fft(q_new)/N, fft(p_new)/N, N, w)
        
        # Fixing 0 mode to constant to avoid division by 0 in LE
        if is_zero_mode_fixed:
            aq_new[0] = a_old[0]
        
        aq_new_cc = np.conj(aq_new)
        a_new = aq_new
        a_new_cc = aq_new_cc

    CLONEs_next_a = []
    CLONEs_next_a_cc = []
    for ct in range(spectrum_num):
        # Evolving clone trajectories
        a_old_clone = CLONEs_old_a[ct]
        a_cc_old_clone = CLONEs_old_a_cc[ct]

        # Evolve a_old_clone
        # TOGGLE CLONE DIRECT a vs. a(q)
        if not is_q_evolution:
            a_new_clone = RK(a_old_clone)
            a_new_cc_clone = np.conj(a_new_clone)
        else:
            Q_old_clone, P_old_clone = from_a_complex_to_Q(a_old_clone)
            
            q_new_clone, p_new_clone = symplectic(ifft(Q_old_clone)*N, ifft(P_old_clone)*N, RHS_super_roll_fdo, w, K)
            aq_new_clone = from_our_Q_to_a_complex_single_step(fft(q_new_clone)/N, fft(p_new_clone)/N, N, w)
            
            # Fixing 0 mode to constant to avoid division by 0 in LE
            if is_zero_mode_fixed:
                aq_new_clone[0] = a_old_clone[0]
            
            aq_new_cc_clone = np.conj(aq_new_clone)
            a_new_clone = aq_new_clone
            a_new_cc_clone = aq_new_cc_clone
        # END TOGGLE CLONE DIRECT a vs. a(q)

        # Append next point in the clones trajectory
        CLONEs_next_a.append(a_new_clone)
        CLONEs_next_a_cc.append(a_new_cc_clone)
        
    CLONEs_2N = []
    u_list = []
    
    if (tt*dt) % T_cycle < 10**(-10) and tt*dt != 0:
        M_a = np.array([a_new, a_new_cc]).reshape(2*N,)
        for ct in range(spectrum_num):
            M_a_clone = np.array([CLONEs_next_a[ct], CLONEs_next_a_cc[ct]]).reshape(2*N,)
            CLONEs_2N.append(M_a_clone)
            
            difference_t = - (M_a - CLONEs_2N[ct])
            v = difference_t
            for gsr in range(ct):
                v = v - np.vdot(difference_t, u_list[gsr]) * u_list[gsr]
            
            # Calculate the normalized vector and the log
            norm_v = np.sqrt(np.real(np.vdot(v, v)))
            if norm_v < 1e-16:
                print(f"Warning: Very small norm for v{ct+1} at iteration {tt}: {norm_v}")
                print('')
                u = v  # Do not normalize if the norm is too small
            else:
                u = v / norm_v
            
            norm_v_scaled = np.sqrt(np.real(np.vdot(np.array(v)/perturbation_of_ic, np.array(v)/perturbation_of_ic)))
            l = np.log(norm_v_scaled + 1e-16)
            
            l_list[ct].append(l)
            u_list.append(u)
            
            if (iteration % cycle < 10**(-10)) and iteration != 0:
                Ly_n = (1/(iteration*T_cycle)) * np.sum(l_list[ct])
                Lyap_spectrum[ct].append(Ly_n)
        
        # Update clones
        CLONEs_old_a = [a_new + u[:N]*perturbation_of_ic for u in u_list]
        CLONEs_old_a_cc = [a_new_cc + u[N:]*perturbation_of_ic for u in u_list]
        
        # Update fiducial trajectory
        a_old = a_new
        
        iteration = iteration + 1
    else:
        a_old = a_new
    
        CLONEs_old_a = CLONEs_next_a
        CLONEs_old_a_cc = CLONEs_next_a_cc
        

############################ VISUALIZING DATA #################################

# DEBUG
fig = plt.figure(figsize=(12,8))

# Calculate the sum of the Lyapunov exponents for each time step
lyapunov_sum = np.sum(Lyap_spectrum, axis=0)

# Plot sum of LE
plt.plot(lyapunov_sum, '-', linewidth=2, color='red', label='Sum of the Lyapunov Exponents')

plt.title(f'a-(N, \u03B1, \u03B2 , \u03B4, T_cycle, dt)=({N},{alpha},{beta},{delta},{T_cycle}, {dt})', fontsize=20)
plt.ylabel('$ \sum_i \lambda_i^{(a)} $', fontsize=22)
plt.xlabel(' n ', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid('on')
plt.legend(loc='best', fontsize='x-large')
plt.show()
 

print('')
print('SIMULATION DATA:')
print('LEs:', Lyap_spectrum)
print('dt:', dt)
print('T_fin:', T_fin)
print('n:', num_iter)
print('T_cycle:', T_cycle)
print('cycle:', cycle)

# Define directory path
folder_name = "LE_pure_symplectic_simulation_results"

# Create directory if needed
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Create file name
if not is_broken:
    file_name = f"pure_sympl_sim_N={N}_alpha={alpha}_beta={beta}_dt={dt}_Tfin={T_fin}_T_cycle={T_cycle}_n={num_iter}_delta={delta}_abs(a0[1])={np.abs(a0[1])}_spectrum_num={spectrum_num}.txt"
else:
    file_name = f"BROKEN_sim_N={N}_alpha={alpha}_beta={beta}_dt={dt}_Tfin={T_fin}_T_cycle={T_cycle}_n={num_iter}_delta={delta}_abs(a0[1])={np.abs(a0[1])}_spectrum_num={spectrum_num}.txt"

# Full file path
file_path = os.path.join(folder_name, file_name)

# Open a file in write mode
with open(file_path, 'w') as file:
    # Write the data to the file
    file.write('\n')
    file.write('SIMULATION DATA:\n')
    file.write(f'LEs: {Lyap_spectrum}\n')
    file.write(f'dt: {dt}\n')
    file.write(f'T_fin: {T_fin}\n')
    file.write(f'n: {num_iter}\n')
    file.write(f'T_cycle: {T_cycle}\n')
    file.write(f'cycle: {cycle}\n')

print('')
print(f"Data saved in '{file_path}'")
  
# Analyze recurrence
if is_analyzing_recurrence and is_q_evolution and not has_antisymmetric_IC:
    from test_FPUT_recurrence_LE import analyze_fput_recurrence
    
    # Execute the complete analysis
    lyap_results, energy_results, analyzer = analyze_fput_recurrence(
        Lyap_spectrum, 
        N=N, alpha=alpha, beta=beta, dt=dt, T_cycle=T_cycle,
        q_trajectory=q_trajectory, 
        p_trajectory=p_trajectory
    )
 

# Defining color sequence
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#c5b0d5', '#e377c2', '#f7b6d2',
                  '#c49c94', '#7f7f7f', '#17becf', '#104E8B', '#AB82FF',
                  '#9400D3', '#bcbd22', '#dbdb8d', '#ff9896', '#9467bd']

# Evolution of the Lyapunov exponents
c_index = 0

fig = plt.figure(figsize=(12,12))
for ll, list_of_lyapunov_exp in enumerate(Lyap_spectrum):
        plt.plot(list_of_lyapunov_exp[:], '-', markersize=1, label=f"\u03BB{ll+1}", color=color_sequence[(c_index + 1)%20])
        c_index += 1

plt.title(f'a-(N, \u03B1, \u03B2 , \u03B4, T_cycle,dt)=({N},{alpha},{beta},{delta},{T_cycle}, {dt})', fontsize=20)
plt.ylabel('$ \lambda_i^{(a)} $',  fontsize=22)
plt.xlabel(' n ', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

text_box = f"is_q_evolution: {is_q_evolution}\nhas_antisymmetric_IC: {has_antisymmetric_IC}\nis_zero_mode_fixed: {is_zero_mode_fixed}"
plt.text(0.98, 0.98, text_box, transform=fig.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.grid('on')
plt.legend(loc='best',fontsize= 'x-large')

plt.show()