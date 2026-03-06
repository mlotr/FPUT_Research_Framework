
"""
FPUT Research Framework
=======================

Script: test_FPUT_recurrence_LE

Performs spectral analysis of recurrence phenomena in the
Fermi–Pasta–Ulam–Tsingou system by comparing the Fourier
spectra of modal energies and Lyapunov exponent evolution.

The goal is to investigate possible relations between
recurrence dynamics and chaotic indicators.

Project README:
https://github.com/mlotr/FPUT_Research_Framework

File descriptions:
https://github.com/mlotr/FPUT_Research_Framework#-file-descriptions

Author: Matteo Lotriglia
Institution: University College Dublin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Plot style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (14, 5)})

# --- Internal Auxiliary Function ---
def _calculate_modal_energies(q_trajectory, p_trajectory, N):
    """Compute the energy of each normal mode at each time step."""
    num_steps = len(q_trajectory)
    all_mode_energies = np.zeros((num_steps, N))
    
    # Angular frequencies of the normal modes (assuming K=m=1)
    k = np.arange(N)
    w_k = np.abs(2 * np.sin(np.pi * k / N))

    for i, (q, p) in enumerate(zip(q_trajectory, p_trajectory)):
        Q_modes = fft(q) / N
        P_modes = fft(p) / N
        energies = 0.5 * (w_k**2 * np.abs(Q_modes)**2 + np.abs(P_modes)**2)
        all_mode_energies[i, :] = energies
        
    return all_mode_energies

# --- Main Function to Call ---
def analyze_fput_recurrence(Lyap_spectrum, N, alpha, beta, dt, T_cycle, 
                              q_trajectory=None, p_trajectory=None, num_modes_to_plot=4):
    """
    Performs a spectral analysis (FFT) to compare the periodicity of 
    Lyapunov exponents and modal energies of the FPUT system.
    
    This version replaces recurrence analysis with frequency-domain analysis.
    The main output is a plot. Return values are kept for 
    compatibility with the calling script.
    
    Args:
        Lyap_spectrum (list): Time series of Lyapunov exponents.
        N (int): Number of particles.
        alpha, beta (float): FPU model parameters (not used in this analysis).
        dt (float): Integration timestep of the simulation.
        T_cycle (float): Time interval between Lyapunov calculations.
        q_trajectory (list): Position trajectory.
        p_trajectory (list): Momentum trajectory.
        num_modes_to_plot (int): Number of modes/exponents to visualize.
    """
    print("\n=== EXECUTING SIMPLIFIED SPECTRAL ANALYSIS (FFT) ===")

    # Check if required data are provided
    if q_trajectory is None or p_trajectory is None:
        print("Error: Trajectory data (q, p) not provided. Cannot proceed.")
        return {}, {}, None

    # 1. Compute the time series of modal energies
    modal_energies_t = _calculate_modal_energies(q_trajectory, p_trajectory, N)

    # 2. Determine how many plots to create and set up the figure
    num_comparisons = min(num_modes_to_plot, len(Lyap_spectrum), N // 2)
    if num_comparisons == 0:
        print("Insufficient data to generate plots.")
        return {}, {}, None

    fig, axes = plt.subplots(num_comparisons, 1, figsize=(14, 4 * num_comparisons), sharex=True)
    if num_comparisons == 1:
        axes = [axes] # Ensures 'axes' is always iterable

    # 3. Iterate over each mode, compute FFTs, and plot results
    for i in range(num_comparisons):
        k = i + 1  # Normal modes are indexed from 1
        ax = axes[i]

        # --- FFT for Lyapunov exponent ---
        lyap_signal = np.array(Lyap_spectrum[i])
        
        if np.mean(lyap_signal) < 1e-9: # Ignore exponents converging to zero/negative
            ax.set_title(f"Exponent $\\lambda_{k}$ negligible. Skipping analysis.")
            continue
            
        N_lyap = len(lyap_signal)
        yf_lyap = fft(lyap_signal)
        xf_lyap = fftfreq(N_lyap, T_cycle)[:N_lyap // 2]
        mag_lyap = 2.0 / N_lyap * np.abs(yf_lyap[0:N_lyap // 2])
        
        ax.plot(xf_lyap, mag_lyap, color='red', lw=2, label=f'Spectrum Exponent $\\lambda_{k}$')

        # --- FFT for Mode Energy ---
        energy_signal = modal_energies_t[:, k]
        N_energy = len(energy_signal)
        yf_energy = fft(energy_signal)
        xf_energy = fftfreq(N_energy, dt)[:N_energy // 2]
        mag_energy = 2.0 / N_energy * np.abs(yf_energy[0:N_energy // 2])
        
        ax.plot(xf_energy, mag_energy, color='blue', alpha=0.8, label=f'Spectrum Mode Energy {k}')
        
        # --- Plot Formatting ---
        ax.set_title(f"FFT Spectrum Comparison: Exponent $\\lambda_{k}$ vs Mode Energy {k}")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4. Finalize and show the plot
    axes[-1].set_xlabel("Frequency [1 / time unit]")
    plt.suptitle("Frequency Spectrum Comparison: Lyapunov vs Modal Energies", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 5. Return placeholders to maintain compatibility with main script
    print("======================================================")
    
    # The main script expects 3 return values.
    # Since the output here is the plot, we return placeholders.
    lyap_results_placeholder = {}
    energy_results_placeholder = {}
    analyzer_placeholder = None
    
    return lyap_results_placeholder, energy_results_placeholder, analyzer_placeholder

