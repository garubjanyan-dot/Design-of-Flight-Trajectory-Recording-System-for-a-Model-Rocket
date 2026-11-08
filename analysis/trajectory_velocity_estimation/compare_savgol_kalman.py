
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import savgol_filter

# =====================================================
# 1. EXPERIMENT CONFIGURATION
# =====================================================

SEED = 42

T = 7.7
FS = 10.0
DT = 1.0 / FS

SIGMA = 0.1

SG_WINDOW = 7
SG_POLYORDER = 2

# Continuous white-jerk noise intensity.
# This is an initial model parameter, not a universal value.
Q_JERK = 0.001

# Ignore initial Kalman convergence period in metrics.
BURN_IN_SECONDS = 1.0

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

t = np.arange(0, T + DT / 2, DT)
N = len(t)


# =====================================================
# 2. TRUE TRAJECTORY AND VELOCITY
# =====================================================

def trajectory(t):
    x = 3.0 * t + 0.2 * t**2
    y = t + 0.1 * t**2
    z = 100.0 + 25.0 * t - 4.905 * t**2

    return np.column_stack((x, y, z))


def analytical_velocity(t):
    vx = 3.0 + 0.4 * t
    vy = 1.0 + 0.2 * t
    vz = 25.0 - 9.81 * t

    return np.column_stack((vx, vy, vz))


position_true = trajectory(t)
velocity_true = analytical_velocity(t)

speed_true = np.linalg.norm(velocity_true, axis=1)


# =====================================================
# 3. GENERATE SYNTHETIC GPS MEASUREMENTS
# =====================================================

noise = rng.normal(
    loc=0.0,
    scale=SIGMA,
    size=(N, 3)
)

position_measured = position_true + noise

print("========== EXPERIMENT CONFIGURATION ==========")
print(f"Duration: {T:.2f} s")
print(f"Sampling frequency: {FS:.1f} Hz")
print(f"Sampling interval: {DT:.2f} s")
print(f"Position samples: {N}")
print(f"Position noise sigma: {SIGMA:.3f} m")


# =====================================================
# 4. METHOD 1: SAVITZKY-GOLAY
# =====================================================

velocity_sg = savgol_filter(
    position_measured,
    window_length=SG_WINDOW,
    polyorder=SG_POLYORDER,
    deriv=1,
    delta=DT,
    axis=0,
    mode="interp"
)


# =====================================================
# 5. METHOD 2: LINEAR KALMAN FILTER
# =====================================================

I = np.eye(3)
Z = np.zeros((3, 3))

# State:
# [x, y, z, vx, vy, vz, ax, ay, az]

F = np.block([
    [I, DT * I, 0.5 * DT**2 * I],
    [Z, I, DT * I],
    [Z, Z, I]
])

# GPS measures position only.
H = np.hstack([
    I,
    np.zeros((3, 6))
])

# Measurement noise covariance.
R = SIGMA**2 * I

# White-jerk process noise covariance.
Q = Q_JERK * np.block([
    [DT**5 / 20 * I, DT**4 / 8 * I, DT**3 / 6 * I],
    [DT**4 / 8 * I, DT**3 / 3 * I, DT**2 / 2 * I],
    [DT**3 / 6 * I, DT**2 / 2 * I, DT * I]
])

# Initial state:
# Position is initialized from the first GPS measurement.
# Initial velocity and acceleration are unknown and set to zero.
state = np.zeros(9)
state[:3] = position_measured[0]

# Large initial uncertainty for velocity and acceleration.
P = np.diag(
    [SIGMA**2] * 3 +
    [100.0] * 3 +
    [100.0] * 3
)

kalman_states = np.zeros((N, 9))
kalman_covariances = np.zeros((N, 9, 9))

# Store initial state.
kalman_states[0] = state
kalman_covariances[0] = P

identity_9 = np.eye(9)

for k in range(1, N):

    # -----------------------------
    # PREDICTION
    # -----------------------------

    state_pred = F @ state

    P_pred = F @ P @ F.T + Q

    # -----------------------------
    # MEASUREMENT UPDATE
    # -----------------------------

    z_k = position_measured[k]

    innovation = z_k - H @ state_pred

    S = H @ P_pred @ H.T + R

    # K = P_pred H.T inv(S), computed by solving
    # a linear system rather than explicitly inverting S.
    K = np.linalg.solve(
        S,
        H @ P_pred
    ).T

    state = state_pred + K @ innovation

    # Joseph form for numerically stable covariance update.
    A = identity_9 - K @ H

    P = (
        A @ P_pred @ A.T
        + K @ R @ K.T
    )

    kalman_states[k] = state
    kalman_covariances[k] = P


velocity_kalman = kalman_states[:, 3:6]
acceleration_kalman = kalman_states[:, 6:9]

speed_kalman = np.linalg.norm(
    velocity_kalman, axis=1
)


# =====================================================
# 6. COMMON EVALUATION TIME INTERVAL
# =====================================================

# SG uses a centered seven-point window.
# Three future samples are required for its estimate.
#
# Kalman starts with unknown velocity and acceleration.
# Exclude first 1 second as an initialization transient.
#
# Compare both methods at identical timestamps.

start_idx = int(round(BURN_IN_SECONDS / DT))
end_idx = N - SG_WINDOW // 2

evaluation_indices = np.arange(start_idx, end_idx)

t_eval = t[evaluation_indices]

v_reference = velocity_true[evaluation_indices]

v_sg_eval = velocity_sg[evaluation_indices]
v_kalman_eval = velocity_kalman[evaluation_indices]

speed_reference = np.linalg.norm(
    v_reference, axis=1
)

speed_sg = np.linalg.norm(v_sg_eval, axis=1)
speed_kalman_eval = np.linalg.norm(v_kalman_eval, axis=1)


# =====================================================
# 7. ERROR METRICS
# =====================================================

def calculate_metrics(reference, estimate):

    error = estimate - reference

    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error**2)),
        "Max_absolute_error": np.max(np.abs(error)),
        "Mean_signed_error": np.mean(error)
    }


methods = {
    "Savitzky-Golay": v_sg_eval,
    "Kalman Filter": v_kalman_eval
}

components = ["vx", "vy", "vz"]

rows = []

for method_name, estimate in methods.items():

    for j, component in enumerate(components):

        result = calculate_metrics(
            v_reference[:, j],
            estimate[:, j]
        )

        result["Method"] = method_name
        result["Quantity"] = component

        rows.append(result)

    # Compare speed magnitude.
    estimated_speed = np.linalg.norm(
        estimate, axis=1
    )

    result = calculate_metrics(
        speed_reference,
        estimated_speed
    )

    result["Method"] = method_name
    result["Quantity"] = "speed"

    rows.append(result)


metrics_df = pd.DataFrame(rows)

print("\n========== METHOD COMPARISON ==========")
print(f"Evaluation interval: {t_eval[0]:.1f} - {t_eval[-1]:.1f} s")
print(f"Evaluation samples: {len(t_eval)}")
print(metrics_df.to_string(index=False))


# =====================================================
# 8. SAVE COMPARISON DATA
# =====================================================

comparison_df = pd.DataFrame({

    "time_s": t_eval,

    "vx_true_m_s": v_reference[:, 0],
    "vy_true_m_s": v_reference[:, 1],
    "vz_true_m_s": v_reference[:, 2],

    "vx_sg_m_s": v_sg_eval[:, 0],
    "vy_sg_m_s": v_sg_eval[:, 1],
    "vz_sg_m_s": v_sg_eval[:, 2],

    "vx_kalman_m_s": v_kalman_eval[:, 0],
    "vy_kalman_m_s": v_kalman_eval[:, 1],
    "vz_kalman_m_s": v_kalman_eval[:, 2],

    "speed_true_m_s": speed_reference,
    "speed_sg_m_s": speed_sg,
    "speed_kalman_m_s": speed_kalman_eval

})

comparison_df.to_csv(
    DATA_DIR / "savgol_kalman_comparison.csv",
    index=False
)

metrics_df.to_csv(
    DATA_DIR / "savgol_kalman_metrics.csv",
    index=False
)


# =====================================================
# 9. PLOT 1: VELOCITY COMPONENTS
# =====================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for j, ax in enumerate(axes):

    ax.plot(
        t_eval,
        v_reference[:, j],
        label="Analytical reference",
        linewidth=2
    )

    ax.plot(
        t_eval,
        v_sg_eval[:, j],
        label="Savitzky-Golay",
        linewidth=1
    )

    ax.plot(
        t_eval,
        v_kalman_eval[:, j],
        label="Kalman Filter",
        linewidth=1
    )

    ax.set_ylabel(f"{components[j]} (m/s)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("Savitzky-Golay vs Kalman: Velocity Components")
fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "savgol_kalman_velocity.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 10. PLOT 2: SPEED MAGNITUDE
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(
    t_eval,
    speed_reference,
    label="Analytical speed",
    linewidth=2
)

plt.plot(
    t_eval,
    speed_sg,
    label="Savitzky-Golay"
)

plt.plot(
    t_eval,
    speed_kalman_eval,
    label="Kalman Filter"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Speed Magnitude Comparison")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "savgol_kalman_speed.png",
    dpi=200
)

plt.close()


# =====================================================
# 11. PLOT 3: VELOCITY ERRORS
# =====================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

for j, ax in enumerate(axes[:3]):

    error_sg = v_sg_eval[:, j] - v_reference[:, j]
    error_kf = v_kalman_eval[:, j] - v_reference[:, j]

    ax.plot(
        t_eval,
        error_sg,
        label="Savitzky-Golay error"
    )

    ax.plot(
        t_eval,
        error_kf,
        label="Kalman error"
    )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel(f"{components[j]} error (m/s)")
    ax.grid(True)
    ax.legend()

error_speed_sg = speed_sg - speed_reference
error_speed_kf = speed_kalman_eval - speed_reference

axes[3].plot(
    t_eval,
    error_speed_sg,
    label="Savitzky-Golay error"
)

axes[3].plot(
    t_eval,
    error_speed_kf,
    label="Kalman error"
)

axes[3].axhline(0, linestyle="--")
axes[3].set_ylabel("Speed error (m/s)")
axes[3].set_xlabel("Time (s)")
axes[3].grid(True)
axes[3].legend()

fig.suptitle("Velocity Estimation Errors")
fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "savgol_kalman_errors.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 12. PLOT 4: KALMAN ACCELERATION ESTIMATE
# =====================================================

true_acceleration = np.tile(
    np.array([0.4, 0.2, -9.81]),
    (N, 1)
)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for j, ax in enumerate(axes):

    ax.plot(
        t,
        true_acceleration[:, j],
        label="True acceleration",
        linewidth=2
    )

    ax.plot(
        t,
        acceleration_kalman[:, j],
        label="Kalman estimate"
    )

    ax.set_ylabel(f"a{j} (m/s²)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("Kalman Acceleration Estimation")
fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "kalman_acceleration.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 13. SAVE ALL KALMAN STATES
# =====================================================

kalman_df = pd.DataFrame({

    "time_s": t,

    "x_est_m": kalman_states[:, 0],
    "y_est_m": kalman_states[:, 1],
    "z_est_m": kalman_states[:, 2],

    "vx_est_m_s": kalman_states[:, 3],
    "vy_est_m_s": kalman_states[:, 4],
    "vz_est_m_s": kalman_states[:, 5],

    "ax_est_m_s2": kalman_states[:, 6],
    "ay_est_m_s2": kalman_states[:, 7],
    "az_est_m_s2": kalman_states[:, 8]

})

kalman_df.to_csv(
    DATA_DIR / "kalman_state_estimates.csv",
    index=False
)


print("\n========== EXPERIMENT COMPLETED ==========")
print(f"Data directory: {DATA_DIR}")
print(f"Plots directory: {PLOTS_DIR}")
print("All input measurements are SYNTHETIC.")
