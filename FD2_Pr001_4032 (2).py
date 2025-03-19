# Erfan Ghodsi
# Mahdi Heydarnejad
# Ali Bahary
# Hedieh Hojaji
# AliAsghar Rezapour
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -----------------------------------------------------------
# 1. Load the data from the text file
# -----------------------------------------------------------
data_file = r"data.txt"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"File {data_file} not found!")

# Define column names as per documentation
column_names = [
    "time",   # Time
    "FAx",    # Aerodynamic force in X direction
    "FAy",    # Aerodynamic force in Y direction
    "FAz",    # Aerodynamic force in Z direction
    "FTx",    # Thrust (or other) force in X direction
    "FTy",    # Thrust force in Y direction
    "FTz",    # Thrust force in Z direction
    "LA",     # Aerodynamic moment about X axis
    "MA",     # Aerodynamic moment about Y axis
    "NA",     # Aerodynamic moment about Z axis
    "LT",     # Additional moment (if defined)
    "MT",     # Additional moment (if defined)
    "NT"      # Additional moment (if defined)
]
df = pd.read_csv(data_file, sep=r"\s+", header=None, names=column_names)

# -----------------------------------------------------------
# 2. Define aircraft parameters and initial flight conditions
# -----------------------------------------------------------
Weight = 564000         # Weight (lbs)
g = 32.2                # Gravitational acceleration (ft/s²)
m = Weight / g          # Mass in slugs
I_XX = 13.7e6           # Moment of inertia about X (slug·ft²)
I_YY = 30.5e6           # Moment of inertia about Y (slug·ft²)
I_ZZ = 43.1e6           # Moment of inertia about Z (slug·ft²)
I_XZ = 0.83e6           # Product of inertia (slug·ft²)

# Initial flight conditions
U0, V0, W0 = 264.6211, 0, 22.2674         # Initial body velocities (ft/s)
X0, Y0, Z0 = 0, 0, 1000                    # Initial positions in inertial frame (ft)
P0, Q0, R0 = 0, 0, 0                       # Initial body rates (rad/s)
phi0, theta0, psi0 = 0, np.deg2rad(1.81), 0  # Initial Euler angles (rad)

# -----------------------------------------------------------
# 3. Integrate linear accelerations (U_dot, V_dot, W_dot) to get body velocities U, V, W
# -----------------------------------------------------------
time = df["time"].values
N = len(time)
dt = np.mean(np.diff(time))  # average time step

# Allocate arrays for body velocities and their derivatives
U = np.zeros(N)
V = np.zeros(N)
W = np.zeros(N)
U_dot_arr = np.zeros(N)
V_dot_arr = np.zeros(N)
W_dot_arr = np.zeros(N)

# Set initial velocities
U[0] = U0
V[0] = V0
W[0] = W0

# In each time step, compute linear accelerations using forces from the data file.
# Here we use fixed initial angular conditions for simplicity.
for i in range(1, N):
    F_A_x = df["FAx"].iloc[i-1]
    F_T_x = df["FTx"].iloc[i-1]
    F_A_y = df["FAy"].iloc[i-1]
    F_T_y = df["FTy"].iloc[i-1]
    F_A_z = df["FAz"].iloc[i-1]
    F_T_z = df["FTz"].iloc[i-1]
    
    U_dot = R0 * V0 - Q0 * W0 - g * np.sin(theta0) + ((F_A_x + F_T_x) / m)
    V_dot = P0 * W0 - R0 * U0 + g * np.cos(theta0) * np.sin(phi0) + ((F_A_y + F_T_y) / m)
    W_dot = U0 * Q0 - P0 * V0 + g * np.cos(theta0) * np.cos(phi0) + ((F_A_z + F_T_z) / m)
    
    U_dot_arr[i] = U_dot
    V_dot_arr[i] = V_dot
    W_dot_arr[i] = W_dot
    
    U[i] = U[i-1] + U_dot * dt
    V[i] = V[i-1] + V_dot * dt
    W[i] = W[i-1] + W_dot * dt

# -----------------------------------------------------------
# 4. Integrate angular accelerations to obtain body rates (P, Q, R)
# -----------------------------------------------------------
P_dot_arr = np.zeros(N)
Q_dot_arr = np.zeros(N)
R_dot_arr = np.zeros(N)
P_int = np.zeros(N)  # integrated body rate P
Q_int = np.zeros(N)  # integrated body rate Q
R_int = np.zeros(N)  # integrated body rate R

P_int[0] = P0
Q_int[0] = Q0
R_int[0] = R0

# For each time step, compute angular accelerations using the moments.
# Solve the 2x2 system for P_dot and R_dot, and compute Q_dot directly.
for i in range(1, N):
    L_total = df["LA"].iloc[i-1] + df["LT"].iloc[i-1]
    M_total = df["MA"].iloc[i-1] + df["MT"].iloc[i-1]
    N_total = df["NA"].iloc[i-1] + df["NT"].iloc[i-1]
    
    b1 = L_total - R0 * Q0 * (I_ZZ - I_YY) + I_XZ * (P0 * Q0)
    b2 = N_total - P0 * Q0 * (I_YY - I_XX) - I_XZ * (R0 * Q0)
    b_vec = np.array([b1, b2])
    A_mat = np.array([[I_XX, -I_XZ],
                      [-I_XZ, I_ZZ]])
    sol = np.linalg.solve(A_mat, b_vec)
    P_dot = sol[0]
    R_dot = sol[1]
    Q_dot = (M_total - P0 * R0 * (I_XX - I_ZZ) - I_XZ * (P0**2 * R0**2)) / I_YY
    
    P_dot_arr[i] = P_dot
    Q_dot_arr[i] = Q_dot
    R_dot_arr[i] = R_dot
    
    P_int[i] = P_int[i-1] + P_dot * dt
    Q_int[i] = Q_int[i-1] + Q_dot * dt
    R_int[i] = R_int[i-1] + R_dot * dt

# -----------------------------------------------------------
# 5. Integrate Euler angles using the nonlinear transformation from body rates to Euler angle rates
# -----------------------------------------------------------
# The transformation from body rates (P, Q, R) to Euler angle derivatives:
#   phi_dot   = P + sin(phi)*tan(theta)*Q + cos(phi)*tan(theta)*R
#   theta_dot = cos(phi)*Q - sin(phi)*R
#   psi_dot   = (sin(phi)/cos(theta))*Q + (cos(phi)/cos(theta))*R
# These are computed at each time step using the updated Euler angles and the integrated body rates.
psi = np.zeros(N)
theta = np.zeros(N)
phi = np.zeros(N)
psi[0] = psi0
theta[0] = theta0
phi[0] = phi0

for i in range(1, N):
    phi_dot = P_int[i-1] + np.sin(phi[i-1]) * np.tan(theta[i-1]) * Q_int[i-1] + np.cos(phi[i-1]) * np.tan(theta[i-1]) * R_int[i-1]
    theta_dot = np.cos(phi[i-1]) * Q_int[i-1] - np.sin(phi[i-1]) * R_int[i-1]
    psi_dot = (np.sin(phi[i-1]) / np.cos(theta[i-1])) * Q_int[i-1] + (np.cos(phi[i-1]) / np.cos(theta[i-1])) * R_int[i-1]
    
    phi[i] = phi[i-1] + phi_dot * dt
    theta[i] = theta[i-1] + theta_dot * dt
    psi[i] = psi[i-1] + psi_dot * dt

# -----------------------------------------------------------
# 6. Compute inertial velocities and integrate to obtain positions (X, Y, Z)
# -----------------------------------------------------------
# To convert from the body frame to the inertial frame, we use the transformation matrix C_b^i.
# The matrix is defined based on Euler angles (phi, theta, psi) as:
#
#   C_b^i = [ [ cosθ*cosψ,   sinϕ*sinθ*cosψ - cosϕ*sinψ,   cosϕ*sinθ*cosψ + sinϕ*sinψ ],
#             [ cosθ*sinψ,   sinϕ*sinθ*sinψ + cosϕ*cosψ,   cosϕ*sinθ*sinψ - sinϕ*cosψ ],
#             [ -sinθ,       sinϕ*cosθ,                  cosϕ*cosθ                  ] ]
#
# Multiply this matrix with the body velocity vector [U, V, W]^T to obtain the inertial velocity vector
# [X_dot, Y_dot, Z_dot]^T. Then integrate using Euler’s method.
X = np.zeros(N)
Y = np.zeros(N)
Z = np.zeros(N)
X_dot_vec = np.zeros(N)
Y_dot_vec = np.zeros(N)
Z_dot_vec = np.zeros(N)

# Set initial positions
X[0] = X0
Y[0] = Y0
Z[0] = Z0

for i in range(0, N):
    ph = phi[i]
    th = theta[i]
    ps = psi[i]
    
    # Build transformation matrix from body to inertial frame
    C_bi = np.array([
        [np.cos(th)*np.cos(ps), np.sin(ph)*np.sin(th)*np.cos(ps) - np.cos(ph)*np.sin(ps), np.cos(ph)*np.sin(th)*np.cos(ps) + np.sin(ph)*np.sin(ps)],
        [np.cos(th)*np.sin(ps), np.sin(ph)*np.sin(th)*np.sin(ps) + np.cos(ph)*np.cos(ps), np.cos(ph)*np.sin(th)*np.sin(ps) - np.sin(ph)*np.cos(ps)],
        [-np.sin(th),           np.sin(ph)*np.cos(th),                                   np.cos(ph)*np.cos(th)]
    ])
    
    # Compute inertial velocity vector by multiplying C_b^i with body velocity vector [U, V, W]^T
    V_body = np.array([U[i], V[i], W[i]])
    V_inertial = C_bi @ V_body  # Matrix multiplication
    
    X_dot_vec[i] = V_inertial[0]
    Y_dot_vec[i] = V_inertial[1]
    Z_dot_vec[i] = V_inertial[2]
    
    if i > 0:
        X[i] = X[i-1] + X_dot_vec[i] * dt
        Y[i] = Y[i-1] + Y_dot_vec[i] * dt
        Z[i] = Z[i-1] + Z_dot_vec[i] * dt

# -----------------------------------------------------------
# 7. Plot the integrated states and 3D trajectory
# -----------------------------------------------------------
# Plot linear states: U, V, W in body frame
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, U, 'r-', label="U (ft/s)")
plt.xlabel("Time (s)")
plt.ylabel("U (ft/s)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time, V, 'g-', label="V (ft/s)")
plt.xlabel("Time (s)")
plt.ylabel("V (ft/s)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(time, W, 'b-', label="W (ft/s)")
plt.xlabel("Time (s)")
plt.ylabel("W (ft/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot integrated body rates: P, Q, R
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, P_int, 'm-', label="P (rad/s)")
plt.xlabel("Time (s)")
plt.ylabel("P (rad/s)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time, Q_int, 'c-', label="Q (rad/s)")
plt.xlabel("Time (s)")
plt.ylabel("Q (rad/s)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(time, R_int, 'y-', label="R (rad/s)")
plt.xlabel("Time (s)")
plt.ylabel("R (rad/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Euler angles: psi, theta, phi
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, psi, 'k-', label=r"$\psi$ (rad)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\psi$ (rad)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time, theta, 'b-', label=r"$\theta$ (rad)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ (rad)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(time, phi, 'r-', label=r"$\phi$ (rad)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\phi$ (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot inertial positions: X, Y, Z (2D plots)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, X, 'r-', label="X (ft)")
plt.xlabel("Time (s)")
plt.ylabel("X (ft)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time, Y, 'g-', label="Y (ft)")
plt.xlabel("Time (s)")
plt.ylabel("Y (ft)")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(time, Z, 'b-', label="Z (ft)")
plt.xlabel("Time (s)")
plt.ylabel("Z (ft)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3D trajectory (X, Y, Z)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, 'r-', label='Trajectory')
ax.set_xlabel("X (ft)")
ax.set_ylabel("Y (ft)")
ax.set_zlabel("Z (ft)")
ax.set_title("3D Trajectory")
ax.legend()
plt.show()