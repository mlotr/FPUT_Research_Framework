"""
Created on Fri Aug  8 10:10:31 2025

@author: matteolotriglia
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from statsmodels.tsa.api import VAR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- DATA ACQUISITION FUNCTIONS ---

def get_financial_data(tickers, start_date, end_date):
    """
    Download adjusted closing price data and calculate logarithmic returns
    (using a HTML session to be more robust to rate limiting).
    """
    session = requests.Session()
    session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    print("Downloading real data...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, session=session)['Close']
    except Exception as e:
        print(f"Error during download: {e}")
        return pd.DataFrame()

    if data.empty or data.isnull().all().all():
        print("ATTENTION: Download failed and produced empty data. Wait before trying again or use synthetic data.")
        return pd.DataFrame()

    print("Download complete.")
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def get_correlated_synthetic_data(tickers, n_obs=1500):
    """
    Generates fake data from known VAR process (with good time structure)
    """
    print("Synthetic data generation with VAR(1) structure...")
    N = len(tickers)
    np.random.seed(42)

    A1 = np.random.rand(N, N) * 3 / N 
    while np.max(np.abs(np.linalg.eigvals(A1))) >= 1.0:
        A1 *= 0.95
        
    intercept = np.random.normal(0.0001, 0.0005, N)
    data = np.zeros((n_obs, N))
    noise_std = 0.02
    
    data[0, :] = np.random.normal(0, noise_std, N)
    
    for t in range(1, n_obs):
        noise = np.random.normal(0, noise_std, N)
        data[t, :] = intercept + A1 @ data[t-1, :] + noise
        
    dates = pd.date_range(start='2020-01-01', periods=n_obs, freq='B')
    log_returns = pd.DataFrame(data, index=dates, columns=tickers)
    print("Generation complete.")
    return log_returns

# --- FUNZIONI DI MODELLAZIONE E EVOLUZIONE ---

def fit_var_model(log_returns):
    """
    Fits VAR model (with control)
    """
    if len(log_returns) < 50:
        raise ValueError(f"Not enough data ({len(log_returns)} obs.) to fit VAR model.")
        
    model = VAR(log_returns)
    fitted_model = model.fit(maxlags=15, ic='aic')
    
    if fitted_model.k_ar == 0:
        raise ValueError("Selected VAR model has 0 lag. Data might be white noise.")
        
    print("\Fitted VAR Model:")
    print(fitted_model.summary())
    return fitted_model

def var_evolution_step(fitted_model, history):
    """
    Single-step evolution using VAR model
    """
    lag_order = fitted_model.k_ar
    forecast_input = history[-lag_order:]
    next_step = fitted_model.forecast(y=forecast_input, steps=1)
    return next_step[0]

###############################
#                             #
#             MAIN            #
#                             #
###############################

# --- 1. SETTINGS ---
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM']
#tickers = ['KO', 'PG', 'JNJ', 'XOM', 'JPM', 'DUK', 'GLD', 'TLT'] # Less correlated portfolio (sectors + asset class)

start_date = '2020-01-01'
end_date = '2024-01-01'
N = len(tickers)

# --- 2. DATA AND MODEL SELECTION ---
#log_returns = get_financial_data(tickers, start_date, end_date)
log_returns = get_correlated_synthetic_data(tickers, n_obs=1500)

if log_returns.empty:
    print("\nThe script cannot proceed because the data has not been loaded.")
else:
    print(f"\nNumber of observations available: {len(log_returns)}")
    print(f"Data shape: {log_returns.shape}")

    fitted_model = fit_var_model(log_returns)
    lag_order = fitted_model.k_ar

    # --- 3. LYAPUNOV SETTINGS ---
    perturbation_of_ic = 1e-3 # Ridotta per stabilità numerica
    spectrum_num = N
    T_cycle = 5
    num_cycles = 200

    print(f"\nCalculating Lyapunov Exponents for {num_cycles} cycles of {T_cycle} days each.")

    # --- 4. PREPARING FIDUCIAL AND CLONE TRAJECTORIES ---
    fiducial_history = log_returns.values[-lag_order:]

    basis_vectors = np.identity(N) * perturbation_of_ic

    clones_history = []
    for i in range(spectrum_num):
        clone_hist = fiducial_history.copy()
        clone_hist[-1, :] += basis_vectors[i, :]
        clones_history.append(clone_hist)

    l_list = [[] for _ in range(spectrum_num)]
    
    # --- 5. EVOLUTION CYCLE ---
    for cycle in tqdm(range(num_cycles)):
        current_fiducial_traj = fiducial_history.copy()
        current_clones_traj = [hist.copy() for hist in clones_history]

        for t in range(T_cycle):
            next_fiducial_step = var_evolution_step(fitted_model, current_fiducial_traj)
            current_fiducial_traj = np.vstack([current_fiducial_traj, next_fiducial_step])

            for i in range(spectrum_num):
                next_clone_step = var_evolution_step(fitted_model, current_clones_traj[i])
                current_clones_traj[i] = np.vstack([current_clones_traj[i], next_clone_step])

        final_fiducial_state = current_fiducial_traj[-1, :]
        final_clones_states = [traj[-1, :] for traj in current_clones_traj]
        diff_vectors = [clone_state - final_fiducial_state for clone_state in final_clones_states]

        u_vectors = []
        for i in range(spectrum_num):
            v = diff_vectors[i]
            for j in range(i):
                proj = np.dot(v, u_vectors[j])
                v = v - proj * u_vectors[j]

            norm_v = np.linalg.norm(v)
            if norm_v > 1e-12:
                u = v / norm_v
                u_vectors.append(u)
                local_lyap = (1 / T_cycle) * np.log(norm_v / perturbation_of_ic)
            else:
                # If a vector collapses, we re-initialize it orthogonal to the others
                # for stability, but the exponent will be very negative.
                u = np.random.randn(N)
                for j in range(len(u_vectors)):
                    u -= np.dot(u, u_vectors[j]) * u_vectors[j]
                u /= np.linalg.norm(u)
                u_vectors.append(u)
                local_lyap = -20 # Very negative value
            
            l_list[i].append(local_lyap)
        
        fiducial_history = current_fiducial_traj[-lag_order:]
        clones_history = []
        for i in range(spectrum_num):
            clone_hist = fiducial_history.copy()
            clone_hist[-1, :] += u_vectors[i] * perturbation_of_ic
            clones_history.append(clone_hist)

    # --- 6. VISUALIZING AND SAVING ---
    Lyap_spectrum_df = pd.DataFrame(l_list).T.replace(-np.inf, -20) # replace inf
    Lyap_spectrum_avg = Lyap_spectrum_df.expanding().mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(spectrum_num):
        ax.plot(Lyap_spectrum_avg.index, Lyap_spectrum_avg[i], label=f'$\\lambda_{{{i+1}}}$ ({tickers[i]})', lw=2)

    mle_avg = Lyap_spectrum_avg.max(axis=1)
    ax.plot(mle_avg.index, mle_avg, label='Maximum LE (MLE)', color='black', lw=3, linestyle='--')
    ax.set_title('Evolution of the Lyapunov Exponents Evolution(FTLEs) of the Portfolio', fontsize=16)
    ax.set_xlabel('Calculation Cycle', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent Average Value', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)
    ax.axhline(0, color='red', linestyle=':', lw=1) # Using 0 as a reference
    
    final_le = Lyap_spectrum_avg.iloc[-1].sort_values(ascending=False)
    print("\nFinal average values of the Lyapunov Exponents:")
    print(final_le)
    
    if final_le.iloc[0] > 0:
        print(f"\nThe system is chaotic (MLE = {final_le.iloc[0]:.4f} > 0).")
        print("A positive value indicates sensitivity to small perturbations (systemic risk).")
    else:
        print(f"\nThe system is stable (MLE = {final_le.iloc[0]:.4f} <= 0).")

    plt.show()

    folder_name = "Financial_Lyapunov_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = f"LE_System_{'_'.join(tickers)}_{end_date}.csv"
    Lyap_spectrum_avg.to_csv(os.path.join(folder_name, file_name))
    print(f"Data saved in '{os.path.join(folder_name, file_name)}'")