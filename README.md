# FPUT Research Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://doi.org/10.1016/j.physd.2025.134813)

A comprehensive computational framework for the analysis of the **Fermi-Pasta-Ulam-Tsingou (FPUT) nonlinear lattice system**, featuring symplectic integration, normal form transformations, and Lyapunov exponent calculations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Authors](#authors)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Architecture](#framework-architecture)
- [File Descriptions](#file-descriptions)
- [Interconnections Between Files](#interconnections-between-files)
- [Example Results](#example-results)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Theoretical References](#theoretical-references)
- [Citation Requirements](#citation-requirements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔬 Overview

This research framework provides advanced numerical tools for studying the FPUT system, a fundamental model in nonlinear dynamics. The framework supports:

- **Two evolution paradigms**: 
  - Direct integration in physical coordinates (q, p) using high-order symplectic integrators
  - Evolution in normal mode coordinates (a) via pre-computed resonance tensors
  
- **Lyapunov spectrum calculation** using the clone method with Gram-Schmidt orthogonalization

- **Tensor-based perturbation theory** up to 5th order (quintet interactions)

- **Comparative analysis** with classical Fermi-Pasta-Ulam formulations

The framework has been developed as part of doctoral research at University College Dublin, combining rigorous mathematical theory with efficient computational implementation.

---

## ✨ Features

- **🚀 High-Performance Integration**
  - Yoshida 8th-order symplectic integrator preserving phase-space structure
  - Optimized FFT-based transformations between coordinate systems
  - Sparse tensor storage for memory efficiency

- **📊 Advanced Analysis Tools**
  - Complete Lyapunov spectrum computation via clone dynamics
  - Spectral (FFT) analysis of recurrence phenomena
  - Modal energy evolution tracking

- **🔧 Flexible Configuration**
  - Support for α-FPUT (quadratic) and β-FPUT (cubic) nonlinearities
  - Customizable initial conditions (harmonic, antisymmetric, etc.)
  - Adjustable system size and perturbation strength

- **📈 Visualization**
  - Automated plotting of Lyapunov exponent evolution
  - Energy distribution analysis
  - Comparative plots (Fermi vs. present formulation)

---

## 👥 Authors

- **Matteo Lotriglia** - University College Dublin (UCD)
  - Doctoral Researcher, School of Mathematics and Statistics
  - GitHub: [@mlot](https://github.com/mlot)
  - Email: matteo.lotriglia@ucdconnect.ie

- **Dr. Tiziana Comito** - UCD
  - Specialist in Hamiltonian Systems

- **Prof. Miguel D. Bustamante** - UCD
  - Thesis Supervisor

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install numpy scipy matplotlib tqdm sortedcontainers colorama
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### Repository Setup

```bash
git clone https://github.com/[username]/fput-research-framework.git
cd fput-research-framework
```

---

## 🚀 Quick Start

### Example 1: Basic Lyapunov Exponent Calculation (q-space evolution)

```python
# Run the main Lyapunov analysis script
python LE_with_pure_sympl_q_evolution.py
```

**Key parameters to modify in the script:**
```python
N = 8                    # Number of particles
alpha = 0.25             # Quadratic nonlinearity coefficient
beta = 0                 # Cubic nonlinearity coefficient
dt = 0.1                 # Time step
T_fin = 30000            # Final integration time
delta = 0.4              # Initial perturbation amplitude
is_q_evolution = True    # Use symplectic q-space integration
```

### Example 2: Pre-compute Interaction Tensors

```python
# Generate resonance tensors for N=16 system
python OPT_tensors_calculator_v02.py
```

This creates a directory `tensors_fermi/tensors_opt_N=16_alpha=1_beta=0.05/` containing sparse tensor files.

### Example 3: Visualize Existing Results

```python
# Plot Lyapunov exponents from saved data
python quick_LE_plotter_v02.py
```

---

## 🏗️ Framework Architecture

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   FPUT Framework                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Initial Conditions] ──┐                               │
│                         │                               │
│                         ├──► [Tensor Pre-computation]   │
│                         │     (OPT_tensors_calculator)  │
│                         │                               │
│                         └──► [Evolution Method Choice]  │
│                                      │                   │
│                         ┌────────────┴────────────┐     │
│                         │                         │     │
│                    [q-space]                 [a-space]  │
│                  (Symplectic)            (RK4 + Tensors)│
│                         │                         │     │
│                         └────────────┬────────────┘     │
│                                      │                   │
│                            [Lyapunov Analysis]          │
│                              (Clone Method)              │
│                                      │                   │
│                         ┌────────────┴────────────┐     │
│                         │                         │     │
│                  [Visualization]          [Data Export] │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Coordinate Systems

The framework operates in three interconnected representations:

1. **Physical coordinates (q, p)**:
   - Real-space particle positions and momenta
   - Evolved via symplectic integrators
   - Natural for initial conditions

2. **Fourier amplitudes (Q, P)**:
   - Complex Fourier coefficients
   - Bridge between q-space and a-space
   - Used in FFT transformations

3. **Normal mode amplitudes (a)**:
   - Complex variables diagonalizing the linear system
   - Related to action-angle variables
   - Ideal for perturbation theory

**Transformations:**
```
q, p  ⟷  Q, P  ⟷  a
     FFT      Normal form
```

---

## 📁 File Descriptions

### Core Numerical Modules

#### `Delta_kronecker.py`
**Purpose**: Generates generalized Kronecker delta tensors for resonance conditions.

**Key Functions**:
- `w_fun(k)`: Computes FPUT dispersion relation ω(k) = 2|sin(πk/N)|
- `delta_k(k, n, resonance_type)`: Creates n-dimensional tensor with 1s where wavenumber conservation holds
- `delta_w(k, n, resonance_type)`: Creates n-dimensional tensor with 1s where both wavenumber AND frequency resonances occur

**Resonance Types** (for n-wave interactions):
- Type 1: `+k₁ -k₂ -k₃ -...` (one positive wavenumber)
- Type 2: `+k₁ +k₂ -k₃ -...` (two positive wavenumbers)
- ...
- Type n: `+k₁ +k₂ +k₃ +...` (all positive)

**Usage Example**:
```python
import Delta_kronecker as d
k = np.arange(0, 16)
delta3_type2 = d.delta_k(k, n=3, resonance_type=2)  # For +ω₁ +ω₂ -ω₃
```

---

#### `OPT_tensors_calculator_v02.py`
**Purpose**: Computes and stores all interaction tensors V, T, A, B, C required for normal form evolution.

**Mathematical Background**:
The FPUT Hamiltonian in normal coordinates is:
```
H = H₀ + H₃ + H₄ + H₅ + ...
```
where:
- **H₀**: Linear oscillator energy
- **H₃**: Triad (3-wave) interactions ∝ α (quadratic nonlinearity)
- **H₄**: Quartet (4-wave) interactions ∝ β (cubic nonlinearity)
- **H₅**: Quintet (5-wave) interactions (from α² and αβ)

**Near-Identity Transformations**:
The code implements canonical transformations:
```
d → c → b → a
```
where each transformation removes non-resonant terms at successive orders.

**Generated Tensors**:

| Tensor | Dimension | Description |
|--------|-----------|-------------|
| `V1, V2, V3` | N×N×N | 3rd-order interaction kernels (triads) |
| `T1, T2, T3, T4` | N×N×N×N | 4th-order interaction kernels (quartets) |
| `A1, A2, A3` | N×N×N | 3rd-order transformation coefficients |
| `B1, B2, B3, B4` | N×N×N×N | 4th-order transformation coefficients |
| `C1, C2, C3, C4, C5` | N×N×N×N×N | 5th-order transformation coefficients |
| `T, W2, W3` | N×N×N×N | Resonant interaction terms |

**Storage Format**:
Tensors are saved in **sparse format** to save space:
- `.npz` files: NumPy compressed format with `{indices, values, shape}`
- `.txt` files: Human-readable list of non-zero entries

**Output Directory Structure**:
```
tensors_fermi/
└── tensors_opt_N=16_alpha=1_beta=0.05/
    ├── V1.npz
    ├── V1.txt
    ├── A1.npz
    ├── A1.txt
    ...
```

**Computational Notes**:
- For N=16, calculation takes ~5-30 minutes depending on hardware
- Memory usage scales as O(N⁵) for quintet tensors
- Progress bars (tqdm) show real-time status

---

#### `OPT_recover_tensors_v02.py`
**Purpose**: Efficient loading of sparse tensors into memory for use in simulations.

**Key Functions**:
- `load_tensor(name, directory)`: Reconstructs full tensor from sparse representation
- `load_all_tensors(N, alpha, beta)`: Loads all tensors for given parameters and creates global variables

**Usage in Main Scripts**:
```python
import OPT_recover_tensors
tensors = OPT_recover_tensors.load_all_tensors(N=16, alpha=1.0, beta=0.05)
# Now V1, A1, B2, etc. are available as global variables
```

**Performance**:
- Loading is much faster than recomputation (seconds vs. minutes)
- Tensors are stored as complex128 arrays in memory

---

### Main Simulation Scripts

#### `LE_with_pure_sympl_q_evolution.py`
**Purpose**: Main simulation script for Lyapunov spectrum calculation.

**Simulation Modes** (toggle with boolean flags):
1. **`is_q_evolution = True`**: 
   - Evolves system in physical coordinates (q, p)
   - Uses Yoshida 8th-order symplectic integrator
   - Recommended for standard FPUT analysis

2. **`is_q_evolution = False`**: 
   - Evolves in normal coordinates (a) using pre-computed tensors
   - Uses 4th-order Runge-Kutta (RK4) split into two half-steps
   - Requires tensors from `OPT_tensors_calculator_v02.py`

**Key Configuration Parameters**:
```python
# System parameters
N = 8                          # Number of oscillators
alpha = 0.25                   # Quadratic nonlinearity
beta = 0                       # Cubic nonlinearity
delta = 0.4                    # Perturbation strength

# Integration parameters
dt = 0.1                       # Time step
T_fin = 30000                  # Final time
T_cycle = 100                  # Lyapunov recalculation interval

# Analysis options
is_q_evolution = True          # Choose evolution method
has_antisymmetric_IC = False   # Use antisymmetric initial conditions
is_zero_mode_fixed = True      # Fix k=0 mode (standard FPUT)
is_analyzing_recurrence = True # Perform FFT recurrence analysis
spectrum_num = 2*N             # Number of Lyapunov exponents
```

**Algorithm: Clone Method for Lyapunov Exponents**

The code implements the Benettin algorithm:

1. **Initialization**:
   - Create fiducial trajectory with initial condition `a₀`
   - Generate `2N` clone trajectories perturbed by `δ = 10⁻⁵`

2. **Evolution Cycle** (every `T_cycle` time units):
   - Evolve fiducial and clone trajectories
   - Compute deviation vectors: `vᵢ = a_clone,ᵢ - a_fiducial`
   - Apply Gram-Schmidt orthonormalization:
     ```
     uᵢ = vᵢ - Σⱼ₌₁^(i-1) (vᵢ · uⱼ) uⱼ
     uᵢ = uᵢ / ||uᵢ||
     ```
   - Accumulate logarithmic growth: `lᵢ(n) = Σ log(||vᵢ||/δ)`
   - Compute Lyapunov exponent: `λᵢ = lᵢ(n) / (n · T_cycle)`
   - Reset clones: `a_clone,ᵢ = a_fiducial + δ · uᵢ`

3. **Output**:
   - Time series of each λᵢ
   - Final Lyapunov spectrum
   - Diagnostic plots
   - Save results to `.txt` file in `LE_pure_symplectic_simulation_results/`

**Symplectic Integrator Details**:

The Yoshida 8th-order integrator uses coefficients:
```python
w3 = 0.784513610477560
w2 = 0.235573213359357
w1 = -1.17767998417887
w0 = 1.315186320683906
```

Update scheme (8 stages per timestep):
```
For r = 1 to 8:
    q ← q + c[r] · dt · p
    p ← p + d[r] · dt · F(q)
```

**Force Calculation**:
```python
def RHS_super_roll_fdo(q):
    q_diff = roll(q, -1) - q  # Forward difference
    F = K·q_diff + α·q_diff² + β·q_diff³
    return F - roll(F, 1)  # Discrete Laplacian
```

**Initial Conditions**:

Standard (Fermi-like):
```python
d₀[k] = δ / (√ω[k] · (N-1) · (1 - 1/N + k/N))  for k=1..N-1
```

Antisymmetric (optional):
```python
q[i] = A·(1 - (i-1)·step)        for i < N/2
q[i] = -q[N-1-i]                 for i > N/2
```

**Output Files**:
```
LE_pure_symplectic_simulation_results/
└── pure_sympl_sim_N=8_alpha=0.25_beta=0_dt=0.1_Tfin=30000.0_..._.txt
```

Contains:
- Full Lyapunov spectrum time series
- Simulation parameters
- Timing information

---

#### `FERMI_vs_US_scale_comparison.py`
**Purpose**: Comparative study between classical Fermi formulation and the present framework.

**Key Differences**:

| Aspect | Fermi Method | Present Framework |
|--------|-------------|-------------------|
| **Boundary Conditions** | Fixed ends: q₀ = q_N = 0 | Periodic: q_N = q₀ |
| **System Size** | N=33 (32 internal oscillators) | N=65 (64 oscillators) |
| **Normal Modes** | Real-valued sine transforms | Complex Fourier modes |
| **Dispersion Relation** | ω(k) = √2 sin(πk/2N) | ω(k) = 2 sin(πk/N) |
| **Energy Definition** | E = T + V separately | E from \|a\|² |

**Analysis Performed**:
1. Evolve both systems with identical parameters (α, β, T, dt)
2. Transform to respective normal coordinates
3. Compute modal energies E_k(t)
4. Calculate energy ratio: R(t) = E_Fermi(t) / E_ours(t)
5. Plot comparison and logarithmic deviations

**Output**:
- Side-by-side energy plots for first 5 modes
- Ratio plots showing convergence/divergence
- Quantitative assessment of formulation equivalence

**Usage**:
```bash
python FERMI_vs_US_scale_comparison.py
```

---

### Analysis and Visualization Tools

#### `test_FPUT_recurrence_LE.py`
**Purpose**: Spectral analysis of recurrence phenomena in FPUT dynamics.

**Theoretical Background**:
The FPUT system exhibits **near-recurrence**: energy initially in low modes returns quasi-periodically. This code investigates whether Lyapunov exponents oscillate at the same frequencies as modal energies.

**Method**:
1. Compute modal energies from (q, p) trajectories:
   ```python
   Q = FFT(q) / N
   P = FFT(p) / N
   E_k = ½(ω_k² |Q_k|² + |P_k|²)
   ```

2. Apply FFT to time series:
   - Lyapunov exponents λᵢ(t)
   - Modal energies E_k(t)

3. Plot amplitude spectra on log scale

4. Compare peak frequencies (resonance identification)

**Key Function**:
```python
def analyze_fput_recurrence(
    Lyap_spectrum,  # List of λᵢ time series
    N, alpha, beta, dt, T_cycle,
    q_trajectory,   # Saved positions
    p_trajectory,   # Saved momenta
    num_modes_to_plot=4
):
    # Computes FFTs and generates comparison plots
    ...
```

**Output**:
- Multi-panel plot showing:
  - Red curves: |FFT(λᵢ)|
  - Blue curves: |FFT(E_k)|
- Identifies shared periodic components

**Interpretation**:
- Matching peaks indicate coupling between chaos (LE) and energy flow
- Absence of peaks suggests stochastic behavior
- Used to validate energy equipartition theories

---

#### `quick_LE_plotter_v02.py`
**Purpose**: Rapid visualization of saved Lyapunov exponent data.

**Features**:
- Automatic parameter extraction from filename using regex
- Plots all λᵢ time series with color-coded curves
- Computes and displays final sum: Σλᵢ (should ≈ 0 for Hamiltonian systems)
- Handles NaN values gracefully

**Usage**:
```python
# Edit file path
file_path = "LE_pure_symplectic_simulation_results/pure_sympl_sim_N=8_..._.txt"
python quick_LE_plotter_v02.py
```

**Output**:
- Single figure with overlaid Lyapunov spectra
- Legend showing λ₁, λ₂, ...
- Title with extracted parameters
- Optional grid and axis customization

**Customization**:
```python
# Uncomment to set y-axis limits
# plt.ylim(-0.01, 0.04)

# Uncomment to save figure
# plt.savefig('plot.png')
```

---

## 🔗 Interconnections Between Files

### Data Flow Diagram

```
[Delta_kronecker.py]
       │
       │ (provides delta functions)
       │
       ├──► [OPT_tensors_calculator_v02.py]
       │            │
       │            │ (generates tensors)
       │            ▼
       │    [tensors_fermi/...]
       │            │
       │            │ (saved to disk)
       │            ▼
       │    [OPT_recover_tensors_v02.py]
       │            │
       │            │ (loads tensors)
       ▼            ▼
[LE_with_pure_sympl_q_evolution.py] ◄──┐
       │                                │
       │ (runs simulation)              │
       │                                │
       ├──► [test_FPUT_recurrence_LE.py]
       │            │
       │            │ (analyzes recurrence)
       │            ▼
       │    [Plots: FFT spectra]
       │
       └──► [LE_pure_symplectic_simulation_results/]
                    │
                    │ (saved data)
                    ▼
            [quick_LE_plotter_v02.py]
                    │
                    ▼
            [Plots: LE evolution]
```

### Typical Workflow

**Scenario A: First-time user, q-space evolution**
```bash
# 1. Run simulation (no tensor pre-computation needed)
python LE_with_pure_sympl_q_evolution.py  # Set is_q_evolution = True

# 2. Visualize results
python quick_LE_plotter_v02.py

# 3. (Optional) Analyze recurrence
# Already done automatically if is_analyzing_recurrence = True
```

**Scenario B: a-space evolution with custom parameters**
```bash
# 1. Pre-compute tensors
python OPT_tensors_calculator_v02.py  # Edit N, alpha, beta

# 2. Run simulation
python LE_with_pure_sympl_q_evolution.py  # Set is_q_evolution = False

# 3. Compare with Fermi formulation
python FERMI_vs_US_scale_comparison.py
```

**Scenario C: Re-analyze existing data**
```bash
# Plot saved Lyapunov exponents
python quick_LE_plotter_v02.py  # Edit file_path
```

---

## 📊 Example Results

### Typical Lyapunov Spectrum (N=8, α=0.25, β=0)

**Observations**:
- **Positive exponents**: λ₁, λ₂, ... indicate exponential divergence (chaos)
- **Zero exponent**: λ_mid ≈ 0 corresponds to energy conservation
- **Negative exponents**: Mirror positive ones (time-reversal symmetry)
- **Sum rule**: Σλᵢ ≈ 0 (phase-space volume preservation)

### Recurrence Spectrum

**Interpretation**:
- Peaks at multiple of fundamental frequency indicate quasi-periodic energy transfer
- Broadband spectrum suggests onset of strong chaos

---

## ⚙️ Advanced Configuration

### Custom Initial Conditions

Modify in `LE_with_pure_sympl_q_evolution.py`:

```python
# Example: Single-mode excitation
a0 = np.zeros(N, dtype=complex)
a0[3] = 0.1 * 1j  # Excite mode k=3

# Example: Two-mode initialization
a0[1] = 0.05
a0[2] = 0.05 * 1j

# Example: Random perturbation
np.random.seed(42)
a0[1:] = (np.random.rand(N-1) + 1j*np.random.rand(N-1)) * delta
```

### Performance Tuning

**For large N (N > 32)**:
1. Use sparse tensor storage (already implemented)
2. Reduce `spectrum_num` if full spectrum not needed:
   ```python
   spectrum_num = N  # Instead of 2*N
   ```
3. Increase `T_cycle` to reduce Gram-Schmidt overhead:
   ```python
   T_cycle = 500  # Instead of 100
   ```

**For long integrations (T > 10⁶)**:
1. Save checkpoints periodically:
   ```python
   if tt % checkpoint_interval == 0:
       np.savez(f'checkpoint_{tt}.npz', a=a_old, Lyap=Lyap_spectrum)
   ```
2. Use smaller `dt` for accuracy:
   ```python
   dt = 0.01  # Instead of 0.1
   ```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1**: `NaN` in Lyapunov exponents
- **Cause**: Numerical instability, timestep too large
- **Solution**: Reduce `dt`, check initial conditions aren't too large

**Issue 2**: Memory error loading tensors
- **Cause**: N too large (N > 32), insufficient RAM
- **Solution**: Use machine with more memory

**Issue 3**: Simulation breaks with "a_new is NaN"
- **Cause**: Overflow in nonlinear terms
- **Solution**: Reduce `delta`, use smaller `dt`, check `alpha` and `beta` values

**Issue 4**: Lyapunov spectrum not converging
- **Cause**: `T_fin` too short, strong chaos
- **Solution**: Increase integration time, ensure `T_cycle` is appropriate

---

## 📚 Theoretical References

**Core Papers**:
1. Comito T., Lotriglia M., Bustamante, M.D. (2025). On the role of 5-wave resonances in the nonlinear dynamics of the Fermi–Pasta–Ulam–Tsingou lattice. *Physica D*, Volume 481.
2. Fermi, E., Pasta, J., & Ulam, S. (1955). Studies of Nonlinear Problems. Los Alamos Report LA-1940.
3. Yoshida, H. (1990). Construction of higher order symplectic integrators. *Physics Letters A*, 150(5-7), 262-268.
4. Benettin, G., et al. (1980). Lyapunov Characteristic Exponents for smooth dynamical systems. *Meccanica*, 15(1), 9-20.
5. Bustamante, M.D., & Kartashova, E. (2009). Effect of the dynamical phases on the nonlinear amplitudes' evolution. *EPL*, 85(1), 14004.


---

## 📜 Citation Requirements

### ⚠️ CITATION POLICY

**Any academic work, research publication, thesis, conference presentation, or technical report that uses this software framework MUST cite the following:**

#### Primary Software Citation:
```bibtex
@software{lotriglia2025fput,
  author       = {Lotriglia, Matteo and Bustamante, Miguel D. and Comito, Tiziana},
  title        = {{FPUT Research Framework: A Computational Toolkit for 
                   Fermi-Pasta-Ulam-Tsingou System Analysis}},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/[username]/fput-research-framework},
  note         = {Accessed: [Date]}
}
```

#### Related Publications (cite when applicable):
```bibtex
@article{COMITO2025134813,
title = {On the role of 5-wave resonances in the nonlinear dynamics of the Fermi-Pasta-Ulam-Tsingou lattice},
journal = {Physica D: Nonlinear Phenomena},
volume = {481},
pages = {134813},
year = {2025},
issn = {0167-2789},
doi = {https://doi.org/10.1016/j.physd.2025.134813},
url = {https://www.sciencedirect.com/science/article/pii/S0167278925002908},
author = {Tiziana Comito and Matteo Lotriglia and Miguel D. Bustamante}
```

### Citation in Publications

**Example acknowledgment text:**
> "Numerical simulations were performed using the FPUT Research Framework developed by Lotriglia et al. (2025) [cite software]. The framework implements high-order symplectic integrators and tensor-based perturbation theory for the analysis of the Fermi-Pasta-Ulam-Tsingou system."

### Non-Academic Use

For commercial or non-academic use, please contact the authors for permission and appropriate citation guidelines.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

### License Summary

- ✅ **You CAN**:
  - Use the software for academic research
  - Modify and extend the code
  - Distribute copies and modifications
  
- ❌ **You CANNOT**:
  - Use without citation (see [Citation Requirements](#citation-requirements))
  - Distribute under a different license
  - Remove author attributions
  
- ⚖️ **You MUST**:
  - Cite the software in any publication using it
  - Keep the same GPL-3.0 license for derivatives