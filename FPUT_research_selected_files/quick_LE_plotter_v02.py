import re
import matplotlib.pyplot as plt
import numpy as np

# File path (example file)
file_path = "LE_pure_symplectic_simulation_results/pure_sympl_sim_N=8_alpha=0.25_beta=0_dt=0.1_Tfin=30000.0_T_cycle=100_n=300000_delta=0.4_abs(a0[1])=0.06531713441797583_spectrum_num=16.txt"

# Extract parameters from file name
N = int(re.search(r'N=(\d+)', file_path).group(1))
alpha = float(re.search(r'alpha=(-?[0-9.]+)', file_path).group(1))
beta = float(re.search(r'beta=(-?[0-9.]+)', file_path).group(1))
delta = float(re.search(r'delta=([0-9.]+)', file_path).group(1))
T_cycle = int(re.search(r'T_cycle=(\d+)', file_path).group(1))
dt = float(re.search(r'dt=([0-9.]+)', file_path).group(1))

# Initialize the list for Lyapunov exponents
Lyap_spectrum = []

# Open and search for LEs
with open(file_path, "r") as file:
    content = file.read()
    
    # Use a regex pattern that matches sequences of complex numbers
    matches = re.search(r"LEs: (\[\[.+\]\])", content)
    #Lyap_spectrum = eval(matches.group(1))
    
    # Avoiding plotting errors due to nan
    if matches:
        # Sostituisci la stringa "nan" con "float('nan')" prima di valutarla
        le_string = matches.group(1).replace("nan", "float('nan')")
        Lyap_spectrum = eval(le_string)
    else:
        print("Pattern 'LEs:' non trovato nel file")
    

# Evolution of the trajectory for one particle
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#c5b0d5', '#e377c2', '#f7b6d2',
                  '#c49c94', '#7f7f7f', '#17becf', '#104E8B', '#AB82FF',
                  '#9400D3', '#bcbd22', '#dbdb8d', '#ff9896', '#9467bd']

c_index = 0

# Original plot
fig = plt.figure(figsize=(12,10))
for ll,list_of_lyapunov_exp in enumerate(Lyap_spectrum):
        plt.plot(np.array(list_of_lyapunov_exp[::]),'-', markersize = 1, label=f"\u03BB{ll+1}", color = color_sequence[(c_index +1)%20])
        c_index += 1

lambda_sum = 0
for ll,list_of_lyapunov_exp in enumerate(Lyap_spectrum):
    lambda_sum += list_of_lyapunov_exp[-1]
    print(list_of_lyapunov_exp[-1])
print("lambda final sum =", lambda_sum)


plt.title(f'a-(N, \u03B1, \u03B2, \u03B5, T_cycle, dt)=({N}, {alpha}, {beta}, {delta}, {T_cycle}, {dt})', fontsize = 20)
plt.ylabel('$ \lambda_i^{(a)} $',  fontsize = 22)
plt.xlabel(' n ', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Set plot limits if needed
# plt.ylim(-0.01, 0.04)
# plt.xlim(10)

plt.grid('on')
ygrid = [0.002*i for i in range(-1, 6)]
plt.yticks()
plt.legend(loc='best',fontsize= 'x-large')

plt.show()
#plt.savefig('plot.png')
