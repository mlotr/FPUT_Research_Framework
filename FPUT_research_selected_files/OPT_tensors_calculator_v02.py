import numpy as np
import warnings
from sortedcontainers import SortedDict  # Sorting dic
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')


# The Dispertion Relation
def w_fun(k):
    '''
    Parameters
    ----------
    k : array-like or scalar
        The funtion requires an array of wavenumbers and
        return the dispersion relation for the FPUT system.

    Returns
    -------
    Omega is an array like / scalar (depending from the input) which
    represent the frequency of the oscillation of the system in linear
    approximation.

    '''
    global K, m, N
    omega = 2 * np.sqrt(K / m) * np.abs(np.sin(pi * (k) / N))
    # omega =2*np.sqrt(K/m)*np.sin(pi*(k)/N)
    # omega = np.round(omega,15)
    
    # Alternative version
    # omega = []
    # for mod in range(0, N):
    #     omega.append(2 * np.abs(np.sin((mod + 1) * np.pi / N)))

    # omega = np.array(omega)

    return (omega)


# %%



def V_123(k1, k2, k3):
    global N, m, K, pi, alpha, w, k

    V0 = np.zeros([N] * 3) + 0j

    coeff = 0j
    coeff2 = 0j
    for i1 in range(1, N):
        for i2 in range(1, N):
            for i3 in range(1, N):
                if (k1[i1] + k2[i2] + k3[i3]) % N == 0:
                    coeff = -8 * 1j * alpha * np.exp(1j * pi * ((k1[i1] + k2[i2] + k3[i3])) / N)
                    coeff2 = 1 / (2 * (m ** (3 / 2)) * (
                        np.sqrt(2 * w[i1] * w[i2] * w[i3])))  # Eq.(16) in the article, m typo in the radicand
                    V0[i1][i2][i3] = coeff2 * coeff * np.sin(pi * k1[i1] / N) * np.sin(pi * k2[i2] / N) * np.sin(
                        pi * k3[i3] / N)
    # MODIF BY MDB2024
    return (V0)


# %%

def T_1234(k1, k2, k3, k4):
    global m, K, alpha, pi, beta, w
    n = len(k1)
    T0 = np.zeros([n] * 4) + 0j
    for i1 in range(1, n):
        for i2 in range(1, n):
            for i3 in range(1, n):
                for i4 in range(1, n):
                    if (k1[i1] + k2[i2] + k3[i3] + k4[i4]) % N == 0:
                        coeff = 16 * beta * np.exp(1j * pi * ((k1[i1] + k2[i2] + k3[i3] + k4[i4])) / N)
                        coeff2 = 1 / (4 * (m ** 2) * (np.sqrt(w[i1] * w[i2] * w[i3] * w[i4])))
                        T0[i1][i2][i3][i4] = coeff * coeff2 * np.sin(pi * k1[i1] / N) * np.sin(
                            pi * k2[i2] / N) * np.sin(pi * k3[i3] / N) * np.sin(pi * k4[i4] / N)
    # MODIF BY MDB2024
    return (T0)


# %%--------------------------------------------------------------------------------------------------------------------------

def A1_123():
    a1 = np.zeros(3 * [len(k)]) * 1j
    for i1 in range(1, N):
        for i2 in range(1, N):
            #                    for i3 in range(1,N):
            i3 = (i1 - i2) % N
            if i3 != 0:

                # MODIF BY MDB2024
                a1[i1][i2][i3] = V1[i1][i2][i3] / (w[i3] + w[i2] - w[i1])

    return (a1)


def A2_123():
    a2 = np.zeros(3 * [len(k)]) * 1j
    for i1 in range(1, N):
        for i2 in range(1, N):
            #                    for i3 in range(1,N):
            i3 = (i1 + i2) % N
            if i3 != 0:

                # MODIF BY MDB2024
                a2[i1][i2][i3] = -2 * V2[i1][i2][i3] / (w[i3] - w[i2] - w[i1])

    return (a2)


def A3_123():
    a3 = np.zeros(3 * [len(k)]) * 1j
    for i1 in range(1, N):
        for i2 in range(1, N):
            #                    for i3 in range(1,N):
            i3 = (-i1 - i2) % N
            if i3 != 0:
                
                # MODIF BY MDB2024
                a3[i1][i2][i3] = -V3[i1][i2][i3] / (-w[i3] - w[i2] - w[i1])

    return (a3)


# %%
# Krasitskii proposed tensors
def B1_1234():
    global T1, k, N, w, eps
    t1 = np.zeros(4 * [len(k)]) * 1j
    B1 = np.zeros(4 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                
                i3 = (i0 - i1 - i2) % N
                if i3 != 0:
                    t1[i0][i1][i2][i3] = (T1[i0][i1][i2][i3] \
                                          + (2 / 3) * (A1[(i2 + i3) % N][i2][i3] * V1[i0][i1][(i0 - i1) % N] \
                                                       + A1[(i1 + i3) % N][i1][i3] * V1[i0][i2][(i0 - i2) % N] \
                                                       + A1[(i1 + i2) % N][i1][i2] * V1[i0][i3][(i0 - i3) % N] \
                                                       + A3[i2][i3][(-i2 - i3) % N] * V1[i1][i0][(i1 - i0) % N] \
                                                       + A3[i1][i3][(-i1 - i3) % N] * V1[i2][i0][(i2 - i0) % N] \
                                                       + A3[i1][i2][(-i2 - i1) % N] * V1[i3][i0][(i3 - i0) % N] \
                                                       ))
                    # MODIF BY MDB2024
                    B1[i0][i1][i2][i3] = - t1[i0][i1][i2][i3] / (- w[i3] - w[i2] - w[i1] + w[i0])  # + 4*eps)
    return (t1, B1)


# %%
def B3_1234():
    global T3
    global k, N, w
    t3 = np.zeros(4 * [len(k)]) * 1j
    B3 = np.zeros(4 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                
                i3 = (i0 + i1 + i2) % N
                if i3 != 0:
                    t3[i0][i1][i2][i3] = (3 * T3[i0][i1][i2][i3] \
                                          + 2 * (A1[(i2 + i1) % N][i1][i2] * V1[i3][i0][(i3 - i0) % N] \
                                                 - A1[i3][i2][(i3 - i2) % N] * V1[(i0 + i1) % N][i0][i1] \
                                                 - A1[i3][i1][(i3 - i1) % N] * V1[(i0 + i2) % N][i0][i2] \
                                                 + A1[i2][i3][(i2 - i3) % N] * V3[i0][i1][(-i0 - i1) % N] \
                                                 + A1[i1][i3][(i1 - i3) % N] * V3[i0][i2][(-i0 - i2) % N] \
                                                 + A3[i1][i2][(-i2 - i1) % N] * V1[i0][i3][(i0 - i3) % N] \
                                                 ))
                    # MODIF BY MDB2024
                    B3[i0][i1][i2][i3] = - t3[i0][i1][i2][i3] / (- w[i3] + w[i2] + w[i1] + w[i0])  # + 4*eps)
    return (t3, B3)


# %%
def B4_1234():
    global T4
    global k, N, w
    t4 = np.zeros(4 * [len(k)]) * 1j
    B4 = np.zeros(4 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                
                i3 = (-i0 - i1 - i2) % N
                if i3 != 0:
                    t4[i0][i1][i2][i3] = (T4[i0][i1][i2][i3] \
                                          + (2 / 3) * (A1[(i2 + i3) % N][i2][i3] * V3[i0][i1][(-i0 - i1) % N] \
                                                       + A1[(i1 + i3) % N][i1][i3] * V3[i0][i2][(-i0 - i2) % N] \
                                                       + A1[(i1 + i2) % N][i1][i2] * V3[i0][i3][(-i0 - i3) % N] \
                                                       - A3[i2][i3][(-i2 - i3) % N] * V1[(i0 + i1) % N][i0][i1] \
                                                       - A3[i1][i3][(-i1 - i3) % N] * V1[(i0 + i2) % N][i0][i2] \
                                                       - A3[i1][i2][(-i1 - i2) % N] * V1[(i0 + i3) % N][i0][i3] \
                                                       ))
                    # MODIF BY MDB2024
                    B4[i0][i1][i2][i3] = -  t4[i0][i1][i2][i3] / (w[i3] + w[i2] + w[i1] + w[i0])  # + 4*eps)
    return (t4, B4)



# %%
def T_Zak():
    '''
    NEW VERSION BY MDB2024

    Returns
    -------
    V2_tilda from Krasitskii.
    V2_tilda = Z + delta_omega*( lambda + Delta)

    '''
    global T2

    Z2 = np.zeros(4 * [len(k)]) * 1j
    Lambda = np.zeros(4 * [len(k)]) * 1j
    B = np.zeros(4 * [len(k)]) * 1j
    lambda_0123 = np.zeros(4 * [len(k)]) * 1j
    T = np.zeros(4 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                
                i3 = (i0 + i1 - i2) % N
                if i3 != 0:
                    Z2[i0][i1][i2][i3] = - 2 * (- A1[i3][i1][(i3 - i1) % N] * V1[i0][i2][(i0 - i2) % N] \
                                                - A1[i1][i3][(i1 - i3) % N] * V1[i2][i0][(i2 - i0) % N] \
                                                - A1[i2][i1][(i2 - i1) % N] * V1[i0][i3][(i0 - i3) % N] \
                                                - A1[i1][i2][(i1 - i2) % N] * V1[i3][i0][(i3 - i0) % N] \
                                                + A1[(i2 + i3) % N][i2][i3] * V1[(i0 + i1) % N][i0][i1] \
                                                - A3[i2][i3][(-i2 - i3) % N] * V3[i0][i1][(-i0 - i1) % N])

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                
                i3 = (i0 + i1 - i2) % N
                if i3 != 0:
                    Delta_0123 = - w[i3] - w[i2] + w[i1] + w[i0]
                    # MODIF BY MDB2024
                    Lambda[i0][i1][i2][i3] = - (A3[i0][i1][(-i0 - i1) % N] * A3[i2][i3][(-i2 - i3) % N] \
                                                + A1[i1][i2][(i1 - i2) % N] * A1[i3][i0][(i3 - i0) % N] \
                                                + A1[i1][i3][(i1 - i3) % N] * A1[i2][i0][(i2 - i0) % N] \
                                                - A1[(i0 + i1) % N][i0][i1] * A1[(i3 + i2) % N][i2][i3] \
                                                - A1[i0][i2][(i0 - i2) % N] * A1[i3][i1][(i3 - i1) % N] \
                                                - A1[i0][i3][(i0 - i3) % N] * A1[i2][i1][(i2 - i1) % N])

                    if (np.abs(Delta_0123) > 10 * eps):
                        
                        lambda_0123[i0][i1][i2][i3] = \
                            - (1 / Delta_0123) * (3 * T2[i0][i1][i2][i3] + (1 / 4) * (
                                    Z2[i0][i1][i2][i3] + Z2[i1][i0][i2][i3] + Z2[i2][i3][i0][i1] + Z2[i3][i2][i0][
                                i1]))

                        B[i0][i1][i2][i3] = Lambda[i0][i1][i2][i3] + lambda_0123[i0][i1][i2][i3]

                        T[i0][i1][i2][
                            i3] = 0  

                    else:
                        
                        lambda_0123[i0][i1][i2][i3] = 0

                        B[i0][i1][i2][i3] = Lambda[i0][i1][i2][i3]

                        T[i0][i1][i2][i3] = 3 * T2[i0][i1][i2][i3] + (1 / 4) * (
                                Z2[i0][i1][i2][i3] + Z2[i1][i0][i2][i3] + Z2[i2][i3][i0][i1] + Z2[i3][i2][i0][i1])

    return (B, T, lambda_0123, Z2)


# %%

def C1_12345():
    C1 = np.zeros(5 * [len(k)]) * 1j
    X1 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 - i1 - i2 - i3) % N
                    if i4 != 0:
                        X1[i0][i1][i2][i3][i4] = (1 / 3) * (1j) * (
                                - V1[i0][(i1 + i2) % N][(i3 + i4) % N] * A1[(i1 + i2) % N][i1][i2] *
                                A1[(i3 + i4) % N][i3][i4] \
                                - V1[i0][(i1 + i3) % N][(i2 + i4) % N] * A1[(i1 + i3) % N][i1][i3] *
                                A1[(i2 + i4) % N][i2][i4] \
                                - V1[i0][(i1 + i4) % N][(i3 + i2) % N] * A1[(i1 + i4) % N][i1][i4] *
                                A1[(i2 + i3) % N][i2][i3] \
                                - V1[(i1 + i2) % N][(-i3 - i4) % N][i0] * A1[(i1 + i2) % N][i1][i2] * A3[i3][i4][
                                    (-i3 - i4) % N] \
                                - V1[(i1 + i3) % N][(-i2 - i4) % N][i0] * A1[(i1 + i3) % N][i1][i3] * A3[i2][i4][
                                    (-i2 - i4) % N] \
                                - V1[(i1 + i4) % N][(-i2 - i3) % N][i0] * A1[(i1 + i4) % N][i1][i4] * A3[i2][i3][
                                    (-i2 - i3) % N] \
                                - V1[(i2 + i3) % N][(-i1 - i4) % N][i0] * A1[(i2 + i3) % N][i2][i3] * A3[i1][i4][
                                    (-i1 - i4) % N] \
                                - V1[(i2 + i4) % N][(-i1 - i3) % N][i0] * A1[(i2 + i4) % N][i2][i4] * A3[i1][i3][
                                    (-i1 - i3) % N] \
                                - V1[(i3 + i4) % N][(-i1 - i2) % N][i0] * A1[(i3 + i4) % N][i3][i4] * A3[i1][i2][
                                    (-i1 - i2) % N] \
                                - (-1) * (V3[i0][(-i1 - i2) % N][(-i3 - i4) % N] * A3[i1][i2][(-i1 - i2) % N] *
                                          A3[i3][i4][(-i3 - i4) % N] \
                                          + V3[i0][(-i1 - i3) % N][(-i2 - i4) % N] * A3[i1][i3][(-i1 - i3) % N] *
                                          A3[i2][i4][(-i2 - i4) % N] \
                                          + V3[i0][(-i1 - i4) % N][(-i2 - i3) % N] * A3[i1][i4][(-i1 - i4) % N] *
                                          A3[i2][i3][(-i2 - i3) % N])) \
                                                 - (1 / 2) * (1j) * (
                                                         T1[i0][i1][i2][(i3 + i4) % N] * A1[(i3 + i4) % N][i3][i4] +
                                                         T1[i0][i1][i3][(i2 + i4) % N] * A1[(i2 + i4) % N][i2][i4] \
                                                         + T1[i0][i1][i4][(i2 + i3) % N] * A1[(i2 + i3) % N][i2][
                                                             i3] + T1[i0][i2][i3][(i1 + i4) % N] *
                                                         A1[(i1 + i4) % N][i1][i4] \
                                                         + T1[i0][i2][i4][(i1 + i3) % N] * A1[(i1 + i3) % N][i1][
                                                             i3] + T1[i0][i3][i4][(i1 + i2) % N] *
                                                         A1[(i1 + i2) % N][i1][i2]) \
                                                 + (1 / 6) * (3 * 1j) * (T2[i1][i2][i0][(-i3 - i4) % N] * A3[i3][i4][
                            (-i3 - i4) % N] + T2[i1][i3][i0][(-i2 - i4) % N] * A3[i2][i4][(-i2 - i4) % N] \
                                                                         + T2[i1][i4][i0][(-i2 - i3) % N] * A3[i2][i3][
                                                                             (-i2 - i3) % N] + T2[i2][i3][i0][
                                                                             (-i1 - i4) % N] * A3[i1][i4][
                                                                             (-i1 - i4) % N] \
                                                                         + T2[i2][i4][i0][(-i1 - i3) % N] * A3[i1][i3][
                                                                             (-i1 - i3) % N] + T2[i3][i4][i0][
                                                                             (-i1 - i2) % N] * A3[i1][i2][
                                                                             (-i1 - i2) % N]) \
                                                 - (1 / 2) * (1j) * (
                                                         V1[i0][i1][(i0 - i1) % N] * B1[(i0 - i1) % N][i2][i3][i4] +
                                                         V1[i0][i2][(i0 - i2) % N] * B1[(i0 - i2) % N][i1][i3][i4] \
                                                         + V1[i0][i3][(i0 - i3) % N] * B1[(i0 - i3) % N][i1][i2][
                                                             i4] + V1[i0][i4][(i0 - i4) % N] *
                                                         B1[(i0 - i4) % N][i1][i2][i3] \
                                                         + (-1) * (V1[i1][i0][(i1 - i0) % N] *
                                                                   B4[(i1 - i0) % N][i2][i3][i4] + V1[i2][i0][
                                                                       (i2 - i0) % N] * B4[(i2 - i0) % N][i1][i3][
                                                                       i4] \
                                                                   + V1[i3][i0][(i3 - i0) % N] *
                                                                   B4[(i3 - i0) % N][i1][i2][i4] + V1[i4][i0][
                                                                       (i4 - i0) % N] * B4[(i4 - i0) % N][i1][i2][
                                                                       i3]))
                        
                        C1[i0][i1][i2][i3][i4] = - (1j) * X1[i0][i1][i2][i3][i4] / (
                                - w[i4] - w[i3] - w[i2] - w[i1] + w[i0])

    return C1


# %%
def X2_12345():
    
    X2 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        # MODIF BY MDB2024
                        X2[i0][i1][i2][i3][i4] = - (4 / 3) * (1j) * (
                                - V1[i0][(i2 + i3) % N][(i4 - i1) % N] * A1[(i2 + i3) % N][i2][i3] * A1[i4][i1][
                            (i4 - i1) % N] \
                                - V1[i0][(i2 + i4) % N][(i3 - i1) % N] * A1[(i2 + i4) % N][i2][i4] * A1[i3][i1][
                                    (i3 - i1) % N] \
                                - V1[i0][(i3 + i4) % N][(i2 - i1) % N] * A1[(i3 + i4) % N][i3][i4] * A1[i2][i1][
                                    (i2 - i1) % N] \
                                - V1[(i2 + i3) % N][(i1 - i4) % N][i0] * A1[i1][i4][(i1 - i4) % N] *
                                A1[(i2 + i3) % N][i2][i3] \
                                - V1[(i2 + i4) % N][(i1 - i3) % N][i0] * A1[i1][i3][(i1 - i3) % N] *
                                A1[(i2 + i4) % N][i2][i4] \
                                - V1[(i4 + i3) % N][(i1 - i2) % N][i0] * A1[i1][i2][(i1 - i2) % N] *
                                A1[(i4 + i3) % N][i3][i4] \
                                - V1[(i2 - i1) % N][(-i3 - i4) % N][i0] * A1[i2][i1][(i2 - i1) % N] * A3[i3][i4][
                                    (-i3 - i4) % N] \
                                - V1[(i3 - i1) % N][(-i2 - i4) % N][i0] * A1[i3][i1][(i3 - i1) % N] * A3[i2][i4][
                                    (-i2 - i4) % N] \
                                - V1[(i4 - i1) % N][(-i2 - i3) % N][i0] * A1[i4][i1][(i4 - i1) % N] * A3[i2][i3][
                                    (-i2 - i3) % N] \
                                + V3[i0][(i1 - i2) % N][(-i3 - i4) % N] * A1[i1][i2][(i1 - i2) % N] * A3[i3][i4][
                                    (-i3 - i4) % N] \
                                + V3[i0][(i1 - i3) % N][(-i2 - i4) % N] * A1[i1][i3][(i1 - i3) % N] * A3[i2][i4][
                                    (-i2 - i4) % N] \
                                + V3[i0][(i1 - i4) % N][(-i2 - i3) % N] * A1[i1][i4][(i1 - i4) % N] * A3[i2][i3][
                                    (-i2 - i3) % N]) \
                                                 + 2 * (1j) * (
                                                         T1[i0][i2][i3][(i4 - i1) % N] * A1[i4][i1][(i4 - i1) % N] +
                                                         T1[i0][i2][i4][(i3 - i1) % N] * A1[i3][i1][(i3 - i1) % N] \
                                                         + T1[i0][i3][i4][(i2 - i1) % N] * A1[i2][i1][
                                                             (i2 - i1) % N] - T1[i2][i1][i0][(-i3 - i4) % N] *
                                                         A3[i3][i4][(-i3 - i4) % N] \
                                                         - T1[i3][i1][i0][(-i2 - i4) % N] * A3[i2][i4][
                                                             (-i2 - i4) % N] - T1[i4][i1][i0][(-i2 - i3) % N] *
                                                         A3[i2][i3][(-i2 - i3) % N]) \
                                                 + (2 / 3) * ((3 * 1j) * T2[i0][i1][i2][(i3 + i4) % N] *
                                                              A1[(i3 + i4) % N][i3][i4] + (3 * 1j) * T2[i0][i1][i3][
                                                                  (i2 + i4) % N] * A1[(i2 + i4) % N][i2][i4] \
                                                              + (3 * 1j) * T2[i0][i1][i4][(i2 + i3) % N] *
                                                              A1[(i2 + i3) % N][i2][i3] - (3 * 1j) * T2[i2][i3][i0][
                                                                  (i1 - i4) % N] * A1[i1][i4][(i1 - i4) % N] \
                                                              - (3 * 1j) * T2[i2][i4][i0][(i1 - i3) % N] * A1[i1][i3][
                                                                  (i1 - i3) % N] - (3 * 1j) * T2[i3][i4][i0][
                                                                  (i1 - i2) % N] * A1[i1][i2][(i1 - i2) % N] \
                                                              - (1j) * T[i3][i4][i1][(i0 - i2) % N] * A1[i0][i2][
                                                                  (i0 - i2) % N] - (1j) * T[i2][i4][i1][(i0 - i3) % N] *
                                                              A1[i0][i3][(i0 - i3) % N] \
                                                              - (1j) * T[i2][i3][i1][(i0 - i4) % N] * A1[i0][i4][
                                                                  (i0 - i4) % N]) \
                                                 + (2 / 3) * (1j) * (- 3 * V1[(i0 + i1) % N][i0][i1] *
                                                                     B1[(i0 + i1) % N][i2][i3][i4] + V1[i0][i2][
                                                                         (i0 - i2) % N] * B2[(i0 - i2) % N][i1][i3][i4] \
                                                                     + V1[i0][i3][(i0 - i3) % N] *
                                                                     B2[(i0 - i3) % N][i1][i2][i4] + V1[i0][i4][
                                                                         (i0 - i4) % N] * B2[(i0 - i4) % N][i1][i2][i3] \
                                                                     - V1[i2][i0][(i2 - i0) % N] *
                                                                     B3[(i2 - i0) % N][i3][i4][i1] - V1[i3][i0][
                                                                         (i3 - i0) % N] * B3[(i3 - i0) % N][i2][i4][i1] \
                                                                     - V1[i4][i0][(i4 - i0) % N] *
                                                                     B3[(i4 - i0) % N][i2][i3][i1] - 3 * V3[i0][i1][
                                                                         (-i0 - i1) % N] * B4[(-i0 - i1) % N][i2][i3][
                                                                         i4])
    # TYPOS CORRECTED BY MDB2024
    return X2


# %%

def X3_12345():
    
    X3 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 + i2 - i3) % N
                    if i4 != 0:
                        # MODIF BY MDB2024
                        X3[i0][i1][i2][i3][i4] = + 2 * (1j) * (
                                - V1[i0][(i3 - i1) % N][(i4 - i2) % N] * A1[i3][i1][(i3 - i1) % N] * A1[i4][i2][
                            (i4 - i2) % N] \
                                - V1[i0][(i4 - i1) % N][(i3 - i2) % N] * A1[i4][i1][(i4 - i1) % N] * A1[i3][i2][
                                    (i3 - i2) % N] \
                                - V1[(i3 - i1) % N][(i2 - i4) % N][i0] * A1[i3][i1][(i3 - i1) % N] * A1[i2][i4][
                                    (i2 - i4) % N] \
                                - V1[(i4 - i1) % N][(i2 - i3) % N][i0] * A1[i4][i1][(i4 - i1) % N] * A1[i2][i3][
                                    (i2 - i3) % N] \
                                - V1[(i3 - i2) % N][(i1 - i4) % N][i0] * A1[i3][i2][(i3 - i2) % N] * A1[i1][i4][
                                    (i1 - i4) % N] \
                                - V1[(i4 - i2) % N][(i1 - i3) % N][i0] * A1[i4][i2][(i4 - i2) % N] * A1[i1][i3][
                                    (i1 - i3) % N] \
                                - V1[(i3 + i4) % N][(i1 + i2) % N][i0] * A1[(i3 + i4) % N][i3][i4] *
                                A1[(i1 + i2) % N][i1][i2] \
                                - V1[i0][(i3 + i4) % N][(-i1 - i2) % N] * A1[(i3 + i4) % N][i3][i4] * A3[i1][i2][
                                    (-i1 - i2) % N] \
                                - V1[(-i1 - i2) % N][(-i3 - i4) % N][i0] * A3[i1][i2][(-i1 - i2) % N] * A3[i3][i4][
                                    (-i3 - i4) % N] \
                                + V3[i0][(i1 - i3) % N][(i2 - i4) % N] * A1[i1][i3][(i1 - i3) % N] * A1[i2][i4][
                                    (i2 - i4) % N] \
                                + V3[i0][(i1 - i4) % N][(i2 - i3) % N] * A1[i1][i4][(i1 - i4) % N] * A1[i2][i3][
                                    (i2 - i3) % N] \
                                + V3[i0][(i1 + i2) % N][(-i3 - i4) % N] * A1[(i1 + i2) % N][i1][i2] * A3[i3][i4][
                                    (-i3 - i4) % N]) \
                                                 - 3 * (1j) * (- T1[i3][i1][i0][(i2 - i4) % N] * A1[i2][i4][
                            (i2 - i4) % N] - T1[i4][i1][i0][(i2 - i3) % N] * A1[i2][i3][(i2 - i3) % N] \
                                                               - T1[i3][i2][i0][(i1 - i4) % N] * A1[i1][i4][
                                                                   (i1 - i4) % N] - T1[i4][i2][i0][(i1 - i3) % N] *
                                                               A1[i1][i3][(i1 - i3) % N] \
                                                               + T1[(i3 + i4) % N][i2][i1][i0] * A1[(i3 + i4) % N][i3][
                                                                   i4] + T1[i0][i3][i4][(-i1 - i2) % N] * A3[i1][i2][
                                                                   (-i1 - i2) % N] \
                                                               - T4[i0][i1][i2][(-i3 - i4) % N] * A3[i3][i4][
                                                                   (-i3 - i4) % N]) \
                                                 + (3 * 1j) * (+ T2[(i1 + i2) % N][i0][i3][i4] * A1[(i1 + i2) % N][i1][
                            i2] - T2[i0][i1][i3][(i4 - i2) % N] * A1[i4][i2][(i4 - i2) % N] \
                                                               - T2[i0][i1][i4][(i3 - i2) % N] * A1[i3][i2][
                                                                   (i3 - i2) % N] - T2[i0][i2][i3][(i4 - i1) % N] *
                                                               A1[i4][i1][(i4 - i1) % N] \
                                                               - T2[i0][i2][i4][(i3 - i1) % N] * A1[i3][i1][
                                                                   (i3 - i1) % N]) + (1j) * (
                                                         T[i3][i4][i2][(i0 + i1) % N] * A1[(i0 + i1) % N][i0][i1] \
                                                         + T[i3][i4][i1][(i0 + i2) % N] * A1[(i0 + i2) % N][i0][
                                                             i2] - T[i1][i2][i4][(i3 - i0) % N] * A1[i3][i0][
                                                             (i3 - i0) % N] \
                                                         - T[i1][i2][i3][(i4 - i0) % N] * A1[i4][i0][(i4 - i0) % N]) \
                                                 + (1j) * (V1[i3][i0][(i3 - i0) % N] * B2[(i3 - i0) % N][i4][i1][i2] +
                                                           V1[i4][i0][(i4 - i0) % N] * B2[(i4 - i0) % N][i3][i1][i2] \
                                                           + V1[(i0 + i1) % N][i0][i1] * B2[(i0 + i1) % N][i2][i3][i4] +
                                                           V1[(i0 + i2) % N][i0][i2] * B2[(i0 + i2) % N][i1][i3][i4] \
                                                           + V3[i0][i1][(-i0 - i1) % N] * B3[(-i0 - i1) % N][i3][i4][
                                                               i2] + V3[i0][i2][(-i0 - i2) % N] *
                                                           B3[(-i0 - i2) % N][i3][i4][i1] \
                                                           - V1[i0][i3][(i0 - i3) % N] * B3[(i0 - i3) % N][i1][i2][i4] -
                                                           V1[i0][i4][(i0 - i4) % N] * B3[(i0 - i4) % N][i1][i2][i3])

                        # CONFIRMED BY MDB2024
    return X3


# %%
def C4_12345():
    C4 = np.zeros(5 * [len(k)]) * 1j
    X4 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 + i2 + i3) % N
                    if i4 != 0:
                        # MODIF BY MDB2024
                        X4[i0][i1][i2][i3][i4] = - (4 / 3) * (1j) * (
                                - V1[(i4 - i1) % N][(i2 + i3) % N][i0] * A1[i4][i1][(i4 - i1) % N] *
                                A1[(i2 + i3) % N][i2][i3] \
                                - V1[(i4 - i2) % N][(i1 + i3) % N][i0] * A1[i4][i2][(i4 - i2) % N] *
                                A1[(i1 + i3) % N][i1][i3] \
                                - V1[(i4 - i3) % N][(i1 + i2) % N][i0] * A1[i4][i3][(i4 - i3) % N] *
                                A1[(i1 + i2) % N][i1][i2] \
                                - V1[(-i1 - i2) % N][(i3 - i4) % N][i0] * A3[i1][i2][(-i1 - i2) % N] * A1[i3][i4][
                                    (i3 - i4) % N] \
                                - V1[(-i1 - i3) % N][(i2 - i4) % N][i0] * A3[i1][i3][(-i1 - i3) % N] * A1[i2][i4][
                                    (i2 - i4) % N] \
                                - V1[(-i2 - i3) % N][(i1 - i4) % N][i0] * A3[i2][i3][(-i2 - i3) % N] * A1[i1][i4][
                                    (i1 - i4) % N] \
                                - V1[i0][(i4 - i1) % N][(-i2 - i3) % N] * A1[i4][i1][(i4 - i1) % N] * A3[i2][i3][
                                    (-i2 - i3) % N] \
                                - V1[i0][(i4 - i2) % N][(-i1 - i3) % N] * A1[i4][i2][(i4 - i2) % N] * A3[i1][i3][
                                    (-i1 - i3) % N] \
                                - V1[i0][(i4 - i3) % N][(-i1 - i2) % N] * A1[i4][i3][(i4 - i3) % N] * A3[i1][i2][
                                    (-i1 - i2) % N] \
                                + V3[i0][(i1 + i2) % N][(i3 - i4) % N] * A1[(i1 + i2) % N][i1][i2] * A1[i3][i4][
                                    (i3 - i4) % N] \
                                + V3[i0][(i1 + i3) % N][(i2 - i4) % N] * A1[(i1 + i3) % N][i1][i3] * A1[i2][i4][
                                    (i2 - i4) % N] \
                                + V3[i0][(i2 + i3) % N][(i1 - i4) % N] * A1[(i2 + i3) % N][i2][i3] * A1[i1][i4][
                                    (i1 - i4) % N]) \
                                                 - 2 * (1j) * (
                                                         T1[i4][i1][i0][(i2 + i3) % N] * A1[(i2 + i3) % N][i2][i3] +
                                                         T1[i4][i2][i0][(i1 + i3) % N] * A1[(i1 + i3) % N][i1][i3] \
                                                         + T1[i4][i3][i0][(i1 + i2) % N] * A1[(i1 + i2) % N][i1][
                                                             i2] - T1[(i4 - i1) % N][i3][i2][i0] * A1[i4][i1][
                                                             (i4 - i1) % N] \
                                                         - T1[(i4 - i2) % N][i3][i1][i0] * A1[i4][i2][
                                                             (i4 - i2) % N] - T1[(i4 - i3) % N][i2][i1][i0] *
                                                         A1[i4][i3][(i4 - i3) % N]) \
                                                 - (2 / 3) * ((3 * 1j) * T2[i0][i1][i4][(-i3 - i2) % N] * A3[i2][i3][
                            (-i2 - i3) % N] + (3 * 1j) * T2[i0][i2][i4][(-i1 - i3) % N] * A3[i1][i3][(-i1 - i3) % N] \
                                                              + (3 * 1j) * T2[i0][i3][i4][(-i1 - i2) % N] * A3[i1][i2][
                                                                  (-i1 - i2) % N] + (1) * (1j) * (
                                                                      - T[i2][i3][i4][(-i0 - i1) % N] * A3[i0][i1][
                                                                  (-i0 - i1) % N] \
                                                                      - T[i1][i3][i4][(-i0 - i2) % N] * A3[i0][i2][
                                                                          (-i0 - i2) % N] - T[i1][i2][i4][
                                                                          (-i0 - i3) % N] * A3[i0][i3][
                                                                          (-i0 - i3) % N])) \
                                                 - 2 * (1j) * (
                                                         T4[i0][i1][i2][(i3 - i4) % N] * A1[i3][i4][(i3 - i4) % N] +
                                                         T4[i0][i1][i3][(i2 - i4) % N] * A1[i2][i4][(i2 - i4) % N] \
                                                         + T4[i0][i2][i3][(i1 - i4) % N] * A1[i1][i4][
                                                             (i1 - i4) % N]) \
                                                 + (2 / 3) * (1j) * (- 3 * V1[i4][i0][(i4 - i0) % N] *
                                                                     B1[(i4 - i0) % N][i1][i2][i3] + (1) * (
                                                                             -V3[i0][i1][(-i0 - i1) % N] *
                                                                             B2[(-i0 - i1) % N][i4][i2][i3] \
                                                                             - V3[i0][i2][(-i0 - i2) % N] *
                                                                             B2[(-i0 - i2) % N][i4][i1][i3] -
                                                                             V3[i0][i3][(-i0 - i3) % N] *
                                                                             B2[(-i0 - i3) % N][i4][i1][i2]) \
                                                                     - V1[(i0 + i1) % N][i0][i1] *
                                                                     B3[(i0 + i1) % N][i2][i3][i4] -
                                                                     V1[(i0 + i2) % N][i0][i2] *
                                                                     B3[(i0 + i2) % N][i1][i3][i4] \
                                                                     - V1[(i0 + i3) % N][i0][i3] *
                                                                     B3[(i0 + i3) % N][i1][i2][i4] + 3 * V1[i0][i4][
                                                                         (i0 - i4) % N] * B4[(i0 - i4) % N][i1][i2][i3])
                        
                        C4[i0][i1][i2][i3][i4] = (1j) * X4[i0][i1][i2][i3][i4] / (
                                -w[i4] + w[i3] + w[i2] + w[i1] + w[i0])

    return C4


# %%
def C5_12345():
    C5 = np.zeros(5 * [len(k)]) * 1j
    X5 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (-i0 - i1 - i2 - i3) % N
                    if i4 != 0:
                        # MODIF BY MDB2024
                        X5[i0][i1][i2][i3][i4] = + (1 / 3) * (1j) * (
                                - V1[(-i1 - i2) % N][(i3 + i4) % N][i0] * A3[i1][i2][(-i1 - i2) % N] *
                                A1[(i3 + i4) % N][i3][i4] \
                                - V1[(-i1 - i3) % N][(i2 + i4) % N][i0] * A3[i1][i3][(-i1 - i3) % N] *
                                A1[(i2 + i4) % N][i2][i4] \
                                - V1[(-i1 - i4) % N][(i2 + i3) % N][i0] * A3[i1][i4][(-i1 - i4) % N] *
                                A1[(i2 + i3) % N][i2][i3] \
                                - V1[(-i2 - i3) % N][(i1 + i4) % N][i0] * A3[i2][i3][(-i2 - i3) % N] *
                                A1[(i1 + i4) % N][i1][i4] \
                                - V1[(-i2 - i4) % N][(i1 + i3) % N][i0] * A3[i2][i4][(-i2 - i4) % N] *
                                A1[(i1 + i3) % N][i1][i3] \
                                - V1[(-i3 - i4) % N][(i1 + i2) % N][i0] * A3[i3][i4][(-i3 - i4) % N] *
                                A1[(i1 + i2) % N][i1][i2] \
                                + V3[i0][(i1 + i2) % N][(i3 + i4) % N] * A1[(i1 + i2) % N][i1][i2] *
                                A1[(i3 + i4) % N][i3][i4] \
                                + V3[i0][(i1 + i3) % N][(i2 + i4) % N] * A1[(i1 + i3) % N][i1][i3] *
                                A1[(i2 + i4) % N][i2][i4] \
                                + V3[i0][(i1 + i4) % N][(i2 + i3) % N] * A1[(i1 + i4) % N][i1][i4] *
                                A1[(i2 + i3) % N][i2][i3] \
                                - V1[i0][(-i1 - i2) % N][(-i3 - i4) % N] * A3[i1][i2][(-i1 - i2) % N] * A3[i3][i4][
                                    (-i3 - i4) % N] \
                                - V1[i0][(-i1 - i3) % N][(-i2 - i4) % N] * A3[i1][i3][(-i1 - i3) % N] * A3[i2][i4][
                                    (-i2 - i4) % N] \
                                - V1[i0][(-i1 - i4) % N][(-i2 - i3) % N] * A3[i1][i4][(-i1 - i4) % N] * A3[i2][i3][
                                    (-i2 - i3) % N]) \
                                                 + (1 / 2) * (1j) * (-T1[(-i1 - i2) % N][i4][i3][i0] * A3[i1][i2][
                            (-i1 - i2) % N] - T1[(-i1 - i3) % N][i4][i2][i0] * A3[i1][i3][(-i1 - i3) % N] \
                                                                     - T1[(-i1 - i4) % N][i3][i2][i0] * A3[i1][i4][
                                                                         (-i1 - i4) % N] - T1[(-i2 - i3) % N][i4][i1][
                                                                         i0] * A3[i2][i3][(-i2 - i3) % N] \
                                                                     - T1[(-i2 - i4) % N][i3][i1][i0] * A3[i2][i4][
                                                                         (-i2 - i4) % N] - T1[(-i3 - i4) % N][i2][i1][
                                                                         i0] * A3[i3][i4][(-i3 - i4) % N] \
                                                                     + T4[i0][i1][i2][(i3 + i4) % N] *
                                                                     A1[(i3 + i4) % N][i3][i4] + T4[i0][i1][i3][
                                                                         (i2 + i4) % N] * A1[(i2 + i4) % N][i2][i4] \
                                                                     + T4[i0][i1][i4][(i2 + i3) % N] *
                                                                     A1[(i2 + i3) % N][i2][i3] + T4[i0][i2][i3][
                                                                         (i1 + i4) % N] * A1[(i1 + i4) % N][i1][i4] \
                                                                     + T4[i0][i2][i4][(i1 + i3) % N] *
                                                                     A1[(i1 + i3) % N][i1][i3] + T4[i0][i3][i4][
                                                                         (i1 + i2) % N] * A1[(i1 + i2) % N][i1][i2] \
                                                                     + V1[(i0 + i1) % N][i0][i1] *
                                                                     B4[(i0 + i1) % N][i2][i3][i4] +
                                                                     V1[(i0 + i2) % N][i0][i2] *
                                                                     B4[(i0 + i2) % N][i1][i3][i4] \
                                                                     + V1[(i0 + i3) % N][i0][i3] *
                                                                     B4[(i0 + i3) % N][i1][i2][i4] +
                                                                     V1[(i0 + i4) % N][i0][i4] *
                                                                     B4[(i0 + i4) % N][i1][i2][i3] \
                                                                     + V3[i0][i1][(-i0 - i1) % N] *
                                                                     B1[(-i0 - i1) % N][i2][i3][i4] + V3[i0][i2][
                                                                         (-i0 - i2) % N] * B1[(-i0 - i2) % N][i1][i3][
                                                                         i4] \
                                                                     + V3[i0][i3][(-i0 - i3) % N] *
                                                                     B1[(-i0 - i3) % N][i1][i2][i4] + V3[i0][i4][
                                                                         (-i0 - i4) % N] * B1[(-i0 - i4) % N][i1][i2][
                                                                         i3])
                        # CONFIRMED BY MDB2024
                        C5[i0][i1][i2][i3][i4] = - (1j) * X5[i0][i1][i2][i3][i4] / (
                                w[i4] + w[i3] + w[i2] + w[i1] + w[i0])  # + 4*eps)

    return C5


# %%

def p_12345_OLD(seq):
    global N

    # seq is a strig. It contains the permutation we want as sequence of numbers divided by commas as '0,1,2,3,4'
    dic_test = {}
    seq = seq.split(',')

    p_ = np.zeros(5 * [len(k)]) * 1j
    temp_0 = np.zeros(5 * [len(k)]) * 1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    for i4 in range(1, N):

                        idx_list = [i0, i1, i2, i3, i4]
                        for seq_idx, seq_el in enumerate(seq):
                            dic_test[seq_el] = idx_list[seq_idx]
                        # dic_test = collections.OrderedDict(sorted(dic_test.items()))
                        dic_test = SortedDict(dic_test)

                        # print(dic_test)
                        temp_0[dic_test[dic_test.iloc[0]]][dic_test[dic_test.iloc[1]]][dic_test[dic_test.iloc[2]]][
                            dic_test[dic_test.iloc[3]]][dic_test[dic_test.iloc[4]]] = A1[i0][i2][(i0 - i2) % N] / (
                                w[(i2 - i0) % N] + w[i3] + w[i4] - w[i1] + 4 * eps)

                        p_[dic_test[dic_test.iloc[0]]][dic_test[dic_test.iloc[1]]][dic_test[dic_test.iloc[2]]][
                            dic_test[dic_test.iloc[3]]][dic_test[dic_test.iloc[4]]] = + (1j) * (1 / 3) * A1[i0][i2][
                            (i0 - i2) % N] * (A1[i3][i1][(i3 - i1) % N] * A1[(i0 - i2) % N][(i3 - i1) % N][i4] \
                                              + A1[i4][i1][(i4 - i1) % N] * A1[(i0 - i2) % N][(i4 - i1) % N][i3] \
                                              + A1[(i3 + i4) % N][i3][i4] * A1[(i3 + i4) % N][(i0 - i2) % N][i1] \
                                              - A3[i3][i4][(-i3 - i4) % N] * A3[(i0 - i2) % N][(-i3 - i4) % N][i1]) \
                                                                                      - (1j) * (2 / 3) * \
                                                                                      temp_0[i0][i1][i2][i3][i4] * (
                                                                                              - A1[i1][i3][
                                                                                                  (i1 - i3) % N] *
                                                                                              V1[(i1 - i3) % N][
                                                                                                  (i2 - i0) % N][i4] \
                                                                                              - A1[i1][i4][
                                                                                                  (i1 - i4) % N] *
                                                                                              V1[(i1 - i4) % N][
                                                                                                  (i2 - i0) % N][i3] \
                                                                                              +
                                                                                              A1[(i3 + i4) % N][i3][
                                                                                                  i4] *
                                                                                              V1[i1][(i2 - i0) % N][
                                                                                                  (i3 + i4) % N] \
                                                                                              + A3[i3][i4][
                                                                                                  (-i3 - i4) % N] *
                                                                                              V1[(i2 - i0) % N][
                                                                                                  (-i3 - i4) % N][
                                                                                                  i1] \
                                                                                              + A1[i3][i1][
                                                                                                  (i3 - i1) % N] *
                                                                                              V3[i4][(i2 - i0) % N][
                                                                                                  (i3 - i1) % N] \
                                                                                              + A1[i4][i1][
                                                                                                  (i4 - i1) % N] *
                                                                                              V3[i3][(i2 - i0) % N][
                                                                                                  (i4 - i1) % N] \
                                                                                              + (3 / 2) *
                                                                                              T1[i1][i3][i4][
                                                                                                  (i2 - i0) % N])

    return p_



# %%
# NEW VERSION BY MDB2024 (ALL TYPOS FIXED)
def p_12345(seq):
    global N

    # seq is a strig. It contains the permutation we want as sequence of numbers divided by commas as '0,1,2,3,4'
    dic_test = {}
    seq = seq.split(',')

    p_ = np.zeros(5 * [len(k)]) * 1j
    
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    for i4 in range(1, N):

                        idx_list = [i0, i1, i2, i3, i4]
                        for seq_idx, seq_el in enumerate(seq):
                            dic_test[seq_el] = idx_list[seq_idx]
                        
                        dic_test = SortedDict(dic_test)

                        p_[dic_test[dic_test.iloc[0]]][dic_test[dic_test.iloc[1]]][dic_test[dic_test.iloc[2]]][
                            dic_test[dic_test.iloc[3]]][dic_test[dic_test.iloc[4]]] = + (1j) * (1 / 3) * A1[i0][i2][
                            (i0 - i2) % N] * (A1[i3][i1][(i3 - i1) % N] * A1[(i0 - i2) % N][(i3 - i1) % N][i4] \
                                              + A1[i4][i1][(i4 - i1) % N] * A1[(i0 - i2) % N][(i4 - i1) % N][i3] \
                                              + A1[(i3 + i4) % N][i3][i4] * A1[(i3 + i4) % N][(i0 - i2) % N][i1] \
                                              - A3[i3][i4][(-i3 - i4) % N] * A3[(i0 - i2) % N][(-i3 - i4) % N][i1]) \
                                                                                      - (1j) * (2 / 3) * A1[i2][i0][
                                                                                          (i2 - i0) % N] / (
                                                                                              w[(i2 - i0) % N] + w[
                                                                                          i3] + w[i4] - w[
                                                                                                  i1] + 4 * eps) * (
                                                                                              - A1[i1][i3][
                                                                                                  (i1 - i3) % N] *
                                                                                              V1[(i1 - i3) % N][
                                                                                                  (i2 - i0) % N][i4] \
                                                                                              - A1[i1][i4][
                                                                                                  (i1 - i4) % N] *
                                                                                              V1[(i1 - i4) % N][
                                                                                                  (i2 - i0) % N][i3] \
                                                                                              +
                                                                                              A1[(i3 + i4) % N][i3][
                                                                                                  i4] *
                                                                                              V1[i1][(i2 - i0) % N][
                                                                                                  (i3 + i4) % N] \
                                                                                              + A3[i3][i4][
                                                                                                  (-i3 - i4) % N] *
                                                                                              V1[(i2 - i0) % N][
                                                                                                  (-i3 - i4) % N][
                                                                                                  i1] \
                                                                                              + A1[i3][i1][
                                                                                                  (i3 - i1) % N] *
                                                                                              V3[i4][(i2 - i0) % N][
                                                                                                  (i3 - i1) % N] \
                                                                                              + A1[i4][i1][
                                                                                                  (i4 - i1) % N] *
                                                                                              V3[i3][(i2 - i0) % N][
                                                                                                  (i4 - i1) % N] \
                                                                                              + (3 / 2) *
                                                                                              T1[i1][i3][i4][
                                                                                                  (i2 - i0) % N])

    return p_


# %%
# Q-FORMULA TYPO FIXED BY MDB2024
def Q_12345(seq):
    global N

    # seq is a strig. It contains the permutation we want as sequence of numbers divided by commas as '0,1,2,3,4'
    dic_test = {}
    seq = seq.split(',')

    Q_ = np.zeros(5 * [len(k)]) * 1j
    #    temp_1 = np.zeros(5*[len(k)])*1j
    #    temp_2 = np.zeros(5*[len(k)])*1j
    #    temp_3 = np.zeros(5*[len(k)])*1j
    #    temp_4 = np.zeros(5*[len(k)])*1j
    #    temp_5 = np.zeros(5*[len(k)])*1j
    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    for i4 in range(1, N):

                        idx_list = [i0, i1, i2, i3, i4]
                        for seq_idx, seq_el in enumerate(seq):
                            dic_test[seq_el] = idx_list[seq_idx]
                        dic_test = SortedDict(dic_test)

                        temp_1 = A1[i2][i4][(i2 - i4) % N] / (w[(i2 - i4) % N] + w[i1] + w[i0] - w[i3] + 4 * eps)
                        temp_2 = A1[i0][i3][(i0 - i3) % N] / (w[(i0 - i3) % N] + w[i1] + w[i2] - w[i4] + 4 * eps)
                        temp_3 = A3[i0][i2][(-i0 - i2) % N] / (w[(i0 + i2) % N] + w[i3] + w[i4] - w[i1] + 4 * eps)
                        temp_4 = A1[(i3 + i4) % N][i3][i4] / (w[(i3 + i4) % N] - w[i0] - w[i1] - w[i2] + 4 * eps)
                        temp_5 = A3[i3][i4][(-i3 - i4) % N] / (w[(i3 + i4) % N] + w[i0] + w[i1] + w[i2] + 4 * eps)

                        Q_[dic_test[dic_test.iloc[0]]][dic_test[dic_test.iloc[1]]][dic_test[dic_test.iloc[2]]][
                            dic_test[dic_test.iloc[3]]][dic_test[dic_test.iloc[4]]] = \
                            (-1j) * A1[i3][i0][(i3 - i0) % N] * (
                                    A1[i2][i4][(i2 - i4) % N] * A1[(i3 - i0) % N][(i2 - i4) % N][i1] \
                                    + (1 / 2) * A1[(i1 + i2) % N][i1][i2] * A1[(i1 + i2) % N][(i3 - i0) % N][i4] \
                                    - (1 / 2) * A3[i1][i2][(-i1 - i2) % N] * A3[i4][(i3 - i0) % N][(-i1 - i2) % N]) \
                            + (-1j) * A1[i4][i2][(i4 - i2) % N] * (
                                    A1[i3][i1][(i3 - i1) % N] * A1[i0][(i4 - i2) % N][(i3 - i1) % N] \
                                    + A3[i0][i1][(-i0 - i1) % N] * A3[i3][(i4 - i2) % N][(-i0 - i1) % N] \
                                    - A1[i1][i3][(i1 - i3) % N] * A1[(i4 - i2) % N][(i1 - i3) % N][i0] \
                                    - A1[i0][i3][(i0 - i3) % N] * A1[(i4 - i2) % N][(i0 - i3) % N][i1]) \
                            + (-1j) * A1[(i0 + i2) % N][i0][i2] * (
                                    A1[i3][i1][(i3 - i1) % N] * A1[(i0 + i2) % N][(i3 - i1) % N][i4] \
                                    + A1[(i3 + i4) % N][i3][i4] * A1[(i3 + i4) % N][(i0 + i2) % N][i1] \
                                    - A1[i1][i3][(i1 - i3) % N] * A1[i4][(i0 + i2) % N][(i1 - i3) % N] \
                                    - A1[i1][i4][(i1 - i4) % N] * A1[i3][(i0 + i2) % N][(i1 - i4) % N] \
                                    - A3[i3][i4][(-i3 - i4) % N] * A3[i1][(i0 + i2) % N][(-i3 - i4) % N]) \
                            - (1j) * (2) * temp_1 * (A1[i0][i3][(i0 - i3) % N] * V3[i1][(i0 - i3) % N][(i2 - i4) % N] \
                                                     + A1[i1][i3][(i1 - i3) % N] * V3[(i1 - i3) % N][(i2 - i4) % N][i0] \
                                                     - A1[i3][i0][(i3 - i0) % N] * V1[(i3 - i0) % N][(i2 - i4) % N][i1] \
                                                     - A1[i3][i1][(i3 - i1) % N] * V1[(i3 - i1) % N][(i2 - i4) % N][i0] \
                                                     + A1[(i0 + i1) % N][i0][i1] * V1[i3][(i0 + i1) % N][(i2 - i4) % N] \
                                                     + A3[i0][i1][(-i0 - i1) % N] * V1[(i2 - i4) % N][(-i0 - i1) % N][
                                                         i3] \
                                                     + (3 / 2) * T1[i3][i0][i1][(i2 - i4) % N]) \
                            + (1j) * (2) * temp_2 * (-A1[i4][i2][(i4 - i2) % N] * V1[(i4 - i2) % N][(i0 - i3) % N][i1] \
                                                     + A1[i2][i4][(i2 - i4) % N] * V3[i1][(i0 - i3) % N][(i2 - i4) % N] \
                                                     + (1 / 2) * A1[(i1 + i2) % N][i1][i2] * V1[i4][(i1 + i2) % N][
                                                         (i0 - i3) % N] \
                                                     + (1 / 2) * A3[i1][i2][(-i1 - i2) % N] *
                                                     V1[(i0 - i3) % N][(-i1 - i2) % N][i4] \
                                                     + (3 / 4) * T1[i4][i1][i2][(i0 - i3) % N]) \
                            + (1j) * (2) * temp_3 * (- A1[i1][i3][(i1 - i3) % N] * V1[(i1 - i3) % N][(-i0 - i2) % N][i4] \
                                                     - A1[i1][i4][(i1 - i4) % N] * V1[(i1 - i4) % N][(-i0 - i2) % N][i3] \
                                                     + A1[i3][i1][(i3 - i1) % N] * V3[i4][(i3 - i1) % N][(-i0 - i2) % N] \
                                                     + A1[i4][i1][(i4 - i1) % N] * V3[i3][(i4 - i1) % N][(-i0 - i2) % N] \
                                                     + A1[(i3 + i4) % N][i3][i4] * V1[i1][(i3 + i4) % N][(-i0 - i2) % N] \
                                                     + A3[i3][i4][(-i3 - i4) % N] * V1[(-i0 - i2) % N][(-i3 - i4) % N][
                                                         i1] \
                                                     + (3 / 2) * T1[i1][i3][i4][(-i0 - i2) % N]) \
                            + (-1j) * (2) * temp_4 * (A1[(i0 + i2) % N][i0][i2] * V1[(i3 + i4) % N][(i0 + i2) % N][i1] \
                                                      + (1 / 2) * A1[(i1 + i2) % N][i1][i2] *
                                                      V1[(i3 + i4) % N][(i1 + i2) % N][i0] \
                                                      + A3[i0][i2][(-i0 - i2) % N] * V1[i1][(i3 + i4) % N][
                                                          (-i0 - i2) % N] \
                                                      + (1 / 2) * A3[i1][i2][(-i1 - i2) % N] * V1[i0][(i3 + i4) % N][
                                                          (-i1 - i2) % N] \
                                                      + (3 / 4) * T1[(i3 + i4) % N][i2][i1][i0]) \
                            - (1j) * (2) * temp_5 * (A1[(i0 + i2) % N][i0][i2] * V3[(-i3 - i4) % N][(i0 + i2) % N][i1] \
                                                     + (1 / 2) * A1[(i1 + i2) % N][i1][i2] *
                                                     V3[(-i3 - i4) % N][(i1 + i2) % N][i0] \
                                                     - A3[i0][i2][(-i0 - i2) % N] * V1[(-i0 - i2) % N][(-i3 - i4) % N][
                                                         i1] \
                                                     - (1 / 2) * A3[i1][i2][(-i1 - i2) % N] *
                                                     V1[(-i1 - i2) % N][(-i3 - i4) % N][i0] \
                                                     + (3 / 4) * T4[i0][i1][i2][(-i3 - i4) % N])

    return Q_


# %%

def W2_12345():
    global X2, C2

    W2_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        # TYPOS FIXED BY MDB2024
                        # MODIF BY MDB2024
                        det = (-w[i4] - w[i3] - w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            W2_[i0][i1][i2][i3][i4] = 0
                        else:
                            W2_[i0][i1][i2][i3][i4] = -(1j) * X2[i0][i1][i2][i3][i4] + det * C2[i0][i1][i2][i3][i4]
                        # DEBUG MDB2024
                        W2_[i0][i1][i2][i3][i4] = -(1j) * X2[i0][i1][i2][i3][i4] + det * C2[i0][i1][i2][i3][i4]
                        # END DEBUG
    return W2_


# %%
# DEF CREATED BY MDB2024

def C2_12345_MDB():
    global X2, C2

    C2_ = np.zeros(5 * [len(k)]) * 1j
    W2_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        
                        det = (-w[i4] - w[i3] - w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            C2_[i0][i1][i2][i3][i4] = (1j) * X2[i0][i1][i2][i3][i4] / det
                            W2_[i0][i1][i2][i3][i4] = 0

                        else:
                            C2_[i0][i1][i2][i3][i4] = C2[i0][i1][i2][i3][i4]
                            W2_[i0][i1][i2][i3][i4] = -(1j) * X2[i0][i1][i2][i3][i4] + det * C2[i0][i1][i2][i3][i4]

    return (C2_, W2_)


# %%
# DEF CREATED BY MDB2024

def C2_12345_lambda_MDB():
    global A1, lambda0123

    C2_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        
                        det = (-w[i4] - w[i3] - w[i2] + w[i1] + w[i0])
                        if (np.abs(det) < 10 * eps):
                            C2_[i0][i1][i2][i3][i4] = (1j) / 2 * (-2 / 3) * (1j) * (
                                    A1[i0][i2][(i0 - i2) % N] * lambda0123[(i0 - i2) % N][i1][i3][i4] \
                                    - A1[i1][i2][(i1 - i2) % N] * lambda0123[(i1 - i2) % N][i0][i3][i4] \
                                    + A1[i0][i3][(i0 - i3) % N] * lambda0123[(i0 - i3) % N][i1][i2][i4] \
                                    - A1[i1][i3][(i1 - i3) % N] * lambda0123[(i1 - i3) % N][i0][i2][i4] \
                                    + A1[i0][i4][(i0 - i4) % N] * lambda0123[(i0 - i4) % N][i1][i2][i3] \
                                    - A1[i1][i4][(i1 - i4) % N] * lambda0123[(i1 - i4) % N][i0][i2][i3])

    return C2_


# %%
# DEF CREATED BY MDB2024

def C2_12345_extra_MDB():
    global X2, C2

    C2_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        
                        det = (-w[i4] - w[i3] - w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            C2_[i0][i1][i2][i3][i4] = (1j) * (
                                    (X2[i0][i1][i2][i3][i4] - X2[i1][i0][i2][i3][i4]) / 2) / det
                        else:
                            C2_[i0][i1][i2][i3][i4] = C2[i0][i1][i2][i3][i4]

    return C2_


# %%

def W3_12345():
    global X3, C3

    W3_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    #                    for i4 in range(1,N):
                    i4 = (i0 + i1 + i2 - i3) % N
                    if i4 != 0:
                        # TYPOS FIXED BY MDB2024
                        # MODIF BY MDB2024
                        det = (-w[i4] - w[i3] + w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            W3_[i0][i1][i2][i3][i4] = 0
                        else:
                            W3_[i0][i1][i2][i3][i4] = (1j) * X3[i0][i1][i2][i3][i4] + det * C3[i0][i1][i2][i3][i4]
                        
                        W3_[i0][i1][i2][i3][i4] = (1j) * X3[i0][i1][i2][i3][i4] + det * C3[i0][i1][i2][i3][i4]

    return W3_


# %%
# DEF CREATED BY MDB2024

def C3_12345_MDB():
    global X3, C3

    C3_ = np.zeros(5 * [len(k)]) * 1j
    W3_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 + i2 - i3) % N
                    if i4 != 0:
                        det = (-w[i4] - w[i3] + w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            C3_[i0][i1][i2][i3][i4] = -(1j) * X3[i0][i1][i2][i3][i4] / det
                            W3_[i0][i1][i2][i3][i4] = 0
                        else:
                            C3_[i0][i1][i2][i3][i4] = C3[i0][i1][i2][i3][i4]
                            W3_[i0][i1][i2][i3][i4] = (1j) * X3[i0][i1][i2][i3][i4] + det * C3[i0][i1][i2][i3][i4]

    return (C3_, W3_)


# %%
# DEF CREATED BY MDB2024

def C3_12345_lambda_MDB():
    global A1, lambda0123

    C3_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 - i2 - i3) % N
                    if i4 != 0:
                        det = (w[i0] + w[i1] - w[i2] - w[i3] - w[i4])
                        if (np.abs(det) < 10 * eps):
                            C3_[i4][i3][i2][i1][i0] = (1j) / 2 * (1j) * (
                                    A1[i0][i2][(i0 - i2) % N] * lambda0123[(i0 - i2) % N][i1][i3][i4] \
                                    + A1[i1][i2][(i1 - i2) % N] * lambda0123[(i1 - i2) % N][i0][i3][i4] \
                                    + A1[i0][i3][(i0 - i3) % N] * lambda0123[(i0 - i3) % N][i1][i2][i4] \
                                    + A1[i1][i3][(i1 - i3) % N] * lambda0123[(i1 - i3) % N][i0][i2][i4] \
                                    - A1[i0][i4][(i0 - i4) % N] * lambda0123[(i0 - i4) % N][i1][i2][i3] \
                                    - A1[i1][i4][(i1 - i4) % N] * lambda0123[(i1 - i4) % N][i0][i2][i3] \
                                    - 2 * A1[(i2 + i4) % N][i4][i2] * lambda0123[(i2 + i4) % N][i3][i0][i1] \
                                    - 2 * A1[(i3 + i4) % N][i4][i3] * lambda0123[(i3 + i4) % N][i2][i0][i1])

    return C3_


# %%
# DEF CREATED BY MDB2024

def C3_12345_extra_MDB():
    global X3, C3

    C3_ = np.zeros(5 * [len(k)]) * 1j

    for i0 in range(1, N):
        for i1 in range(1, N):
            for i2 in range(1, N):
                for i3 in range(1, N):
                    
                    i4 = (i0 + i1 + i2 - i3) % N
                    if i4 != 0:
                        det = (-w[i4] - w[i3] + w[i2] + w[i1] + w[i0])
                        if (np.abs(det) > 10 * eps):
                            C3_[i0][i1][i2][i3][i4] = -(1j) * ((X3[i0][i1][i2][i3][i4] - 3 / 2 * X2[i4][i3][i2][i1][
                                i0] + X3[i0][i1][i2][i4][i3] - 3 / 2 * X2[i3][i4][i2][i1][i0]) / 2) / det
                        else:
                            C3_[i0][i1][i2][i3][i4] = C3[i0][i1][i2][i3][i4]

    return C3_


# %%


#######################################

# MAIN
pi = np.pi
eps = np.finfo(float).eps

N = 16  # number of particle
L = 2 * pi  # domain lenght
dx = L / N  # step size
x = np.arange(0, L, dx)  # 1D space grid

# Costants
K = 1  # linear coeff
beta = 0.05  # Cubic non-line#ar coeff
alpha = 1  # quadratic non-linearity coeff
m = 1  # mass of the particles

k = np.arange(0, N, 1)
w = w_fun(k)

# CALCULATING TENSORS

# Equations kernels
V1 = V_123(-k, k, k)
V2 = V_123(k, k, -k)
V3 = V_123(k, k, k)
T1 = T_1234(-k, k, k, k)
T2 = T_1234(-k, -k, k, k)
T3 = T_1234(k, k, k, -k)
T4 = T_1234(k, k, k, k)

# Transformation's coefficients 3-waves and 4-waves
A1 = A1_123()
A2 = A2_123()
A3 = A3_123()
t1, B1 = B1_1234()
t3, B3 = B3_1234()
t4, B4 = B4_1234()

# Zakharov tensor with 'little lambda' modification MDB2024
B2, T, lambda0123, t2 = T_Zak()

# Transformation's coefficients 5-waves
P01234 = p_12345('0,1,2,3,4') + p_12345('0,1,3,2,4') + p_12345('0,1,4,2,3')
P10234 = p_12345('1,0,2,3,4') + p_12345('1,0,3,2,4') + p_12345('1,0,4,2,3')
C1 = C1_12345()
X2 = X2_12345()
X3 = X3_12345()

C2_lambda = C2_12345_lambda_MDB()
C2_P = (-1j) * (P01234 - P10234)
C2 = C2_P + C2_lambda

# Resonant 5-waves interaction kernels
C2, W2 = C2_12345_MDB()
C3_lambda = C3_12345_lambda_MDB()
C3_Q = (1j) * (1 / 2) * (Q_12345('0,1,2,3,4') + Q_12345('0,2,1,3,4') + Q_12345('0,1,2,4,3') + Q_12345('0,2,1,4,3'))
C3 = C3_Q + C3_lambda
C3, W3 = C3_12345_MDB()
C4 = C4_12345()
C5 = C5_12345()


# SAVING
directory = f"tensors_fermi/tensors_opt_N={N}_alpha={alpha}_beta={beta}"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

def save_tensor(name, tensor):
    """
    Save a tensor by storing only its non-zero entries.
    Saves both a .txt file (human readable) and a .npz file (efficient numpy format)
    
    Args:
        name: Name of the tensor
        tensor: NumPy array to save
    """
    # Get non-zero entries and their indices
    non_zero_indices = np.nonzero(tensor)
    values = tensor[non_zero_indices]
    
    # Save shape information and non-zero entries in NPZ format
    np.savez(os.path.join(directory, f"{name}.npz"),
             shape=tensor.shape,
             indices=non_zero_indices,
             values=values)
    
    # Save human-readable version in TXT format
    with open(f"{directory}/{name}.txt", "w") as file:
        file.write(f"Shape: {tensor.shape}\n")
        file.write("Non-zero entries (indices: value):\n")
        for idx, val in zip(zip(*non_zero_indices), values):
            file.write(f"{idx}: {val}\n")

def save_all_tensors(tensors_dict):
    """
    Save multiple tensors in sparse format.
    
    Args:
        tensors_dict: Dictionary mapping tensor names to tensor objects
    """
    print("Saving tensors in sparse format...")
    for name, tensor in tqdm(tensors_dict.items()):
        save_tensor(name, tensor)
    print("All tensors saved successfully!")
   

# Save tensors on files with specific names
tensors = {
    "V1": V1, "V2": V2, "V3": V3,
    "T1": T1, "T2": T2, "T3": T3, "T4": T4,
    "A1": A1, "A2": A2, "A3": A3,
    "B1": B1,"B2": B2, "B3": B3, "B4": B4,
    "T": T, "W2": W2, "W3": W3,
    "C1": C1, "X2": X2, "X3": X3,
    "C2_lambda": C2_lambda, "C2_P": C2_P, "C2": C2,
    "C3_lambda": C3_lambda, "C3_Q": C3_Q, "C3": C3,
    "C4": C4, "C5": C5
}
    
save_all_tensors(tensors)





