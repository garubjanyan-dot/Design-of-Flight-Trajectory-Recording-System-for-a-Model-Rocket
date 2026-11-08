
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import savgol_filter

# =====================================================
# 1. EXPERIMENT PARAMETERS
# =====================================================

SEED = 42
T = 7.7
FS = 10.0
DT = 1.0 / FS

SIGMA = 0.1

SG_WINDOWS = [7, 11, 15, 21]
SG_POLYORDER = 2

Q_JERK = 0.001
BURN_IN = 1.0

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

t = np.arange(0, T + DT / 2, DT)
N = len(t)


# =====================================================
# 2. TRUE MATHEMATICAL MODEL
# =====================================================

def trajectory(t):

    x = 3.0 * t + 0.2 * t**2
    y = 1.0 * t + 0.1 * t**2
    z = 100.0 + 25.0 * t - 4.905 * t**2

    return np.column_stack((x, y, z))


def analytical_velocity(t):

    vx = 3.0 + 0.4 * t
    vy = 1.0 + 0.2 * t
    vz = 25.0 - 9.81 * t

    return np.column_stack((vx, vy, vz))


position_true = trajectory(t)


# =====================================================
# 3. GENERATE SYNTHETIC GPS MEASUREMENTS
# =====================================================

noise = rng.normal(
    0.0,
    SIGMA,
    size=(N, 3)
)

position_measured = position_true + noise

velocity_true = analytical_velocity(t)


# =====================================================
# 4. SAVITZKY-GOLAY FOR DIFFERENT WINDOWS
# =====================================================

sg_results = {}

for window in SG_WINDOWS:

    sg_results[window] = savgol_filter(
        position_measured,
        window_length=window,
        polyorder=SG_POLYORDER,
        deriv=1,
        delta=DT,
        axis=0,
        mode="interp"
    )


# =====================================================
# 5. LINEAR KALMAN FILTER
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

# Measurement covariance.

R = SIGMA**2 * I

# White-jerk process noise covariance.

Q = Q_JERK * np.block([
    [DT**5 / 20 * I, DT**4 / 8 * I, DT**3 / 6 * I],
    [DT**4 / 8 * I, DT**3 / 3 * I, DT**2 / 2 * I],
    [DT**3 / 6 * I, DT**2 / 2 * I, DT * I]
])

# Initial state.

state = np.zeros(9)
state[:3] = position_measured[0]

P = np.diag(
    [SIGMA**2] * 3 +
    [100.0] * 3 +
    [100.0] * 3
)

kalman_states = np.zeros((N, 9))
kalman_states[0] = state

I9 = np.eye(9)

for k in range(1, N):

    # -----------------------------
    # PREDICTION
    # -----------------------------

    state_pred = F @ state
    P_pred = F @ P @ F.T + Q

    # -----------------------------
    # UPDATE
    # -----------------------------

    z_k = position_measured[k]

    innovation = z_k - H @ state_pred
    S = H @ P_pred @ H.T + R

    K = np.linalg.solve(
        S,
        H @ P_pred
    ).T

    state = state_pred + K @ innovation

    # Joseph covariance update.

    A = I9 - K @ H

    P = (
        A @ P_pred @ A.T
        + K @ R @ K.T
    )

    kalman_states[k] = state


velocity_kalman = kalman_states[:, 3:6]


# =====================================================
# 6. COMMON EVALUATION TIMESTAMPS
# =====================================================

# Largest window = 21, half-width = 10 samples.
# Burn-in = 1 second = 10 samples.

max_half_window = max(
    (w - 1) // 2 for w in SG_WINDOWS
)

burn_in_index = int(round(BURN_IN / DT))

start_idx = max(
    max_half_window,
    burn_in_index
)

end_idx = N - max_half_window

indices = np.arange(start_idx, end_idx)

t_eval = t[indices]

velocity_reference = analytical_velocity(t_eval)

# Kalman evaluated at exactly the same timestamps.

kalman_eval = velocity_kalman[indices]


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


metrics_rows = []

for window in SG_WINDOWS:

    estimate = sg_results[window][indices]

    for j, component in enumerate(["vx", "vy", "vz"]):

        result = calculate_metrics(
            velocity_reference[:, j],
            estimate[:, j]
        )

        metrics_rows.append({
            "Method": "Savitzky-Golay",
            "Window": window,
            "Quantity": component,
            **result
        })

    # Speed magnitude.

    true_speed = np.linalg.norm(
        velocity_reference, axis=1
    )

    estimated_speed = np.linalg.norm(
        estimate, axis=1
    )

    result = calculate_metrics(
        true_speed,
        estimated_speed
    )

    metrics_rows.append({
        "Method": "Savitzky-Golay",
        "Window": window,
        "Quantity": "speed",
        **result
    })


# Kalman metrics on exactly the same timestamps.

for j, component in enumerate(["vx", "vy", "vz"]):

    result = calculate_metrics(
        velocity_reference[:, j],
        kalman_eval[:, j]
    )

    metrics_rows.append({
        "Method": "Kalman",
        "Window": np.nan,
        "Quantity": component,
        **result
    })


true_speed = np.linalg.norm(
    velocity_reference, axis=1
)

kalman_speed = np.linalg.norm(
    kalman_eval, axis=1
)

result = calculate_metrics(
    true_speed,
    kalman_speed
)

metrics_rows.append({
    "Method": "Kalman",
    "Window": np.nan,
    "Quantity": "speed",
    **result
})


metrics_df = pd.DataFrame(metrics_rows)


# =====================================================
# 8. VECTOR RMSE
# =====================================================

vector_rows = []

for window in SG_WINDOWS:

    estimate = sg_results[window][indices]

    error = estimate - velocity_reference

    vector_rmse = np.sqrt(
        np.mean(np.sum(error**2, axis=1))
    )

    vector_rows.append({
        "Method": "Savitzky-Golay",
        "Window": window,
        "Vector_RMSE": vector_rmse
    })


kalman_error = kalman_eval - velocity_reference

kalman_vector_rmse = np.sqrt(
    np.mean(np.sum(kalman_error**2, axis=1))
)

vector_rows.append({
    "Method": "Kalman",
    "Window": np.nan,
    "Vector_RMSE": kalman_vector_rmse
})

vector_df = pd.DataFrame(vector_rows)


# =====================================================
# 9. PRINT RESULTS
# =====================================================

print("\n========== EXPERIMENT ==========")

print(f"Position samples: {N}")
print(f"Evaluation samples: {len(indices)}")
print(f"Evaluation interval: {t_eval[0]:.1f} - {t_eval[-1]:.1f} s")

print("\n========== METRICS ==========")

print(
    metrics_df.round(6).to_string(index=False)
)

print("\n========== 3D VECTOR RMSE ==========")

print(
    vector_df.round(6).to_string(index=False)
)

print("\n========== SG THEORETICAL NOISE ==========")

for window in SG_WINDOWS:

    m = (window - 1) // 2

    sum_j_squared = sum(
        j**2 for j in range(-m, m + 1)
    )

    sigma_velocity = (
        SIGMA / (DT * np.sqrt(sum_j_squared))
    )

    print(
        f"Window {window}: "
        f"{sigma_velocity:.6f} m/s"
    )


# =====================================================
# 10. SAVE CSV FILES
# =====================================================

metrics_df.to_csv(
    DATA_DIR / "sg_window_metrics.csv",
    index=False
)

vector_df.to_csv(
    DATA_DIR / "sg_window_vector_rmse.csv",
    index=False
)


# Detailed comparison for SG window 11 and Kalman.

sg11 = sg_results[11][indices]


# =====================================================
#  EXTRA EXPERIMENT: SG + KALMAN ESTIMATOR FUSION
# =====================================================

sg_estimate = sg_results[11][indices]
kf_estimate = kalman_eval

# Errors relative to analytical velocity
error_sg = sg_estimate - velocity_reference
error_kf = kf_estimate - velocity_reference

components = ["vx", "vy", "vz"]

combined_velocity = np.zeros_like(velocity_reference)

print("\n========== ERROR CORRELATION ==========")

for j, component in enumerate(components):

    # Error variances
    var_sg = np.var(error_sg[:, j], ddof=1)
    var_kf = np.var(error_kf[:, j], ddof=1)

    # Error covariance
    covariance = np.cov(
        error_sg[:, j],
        error_kf[:, j]
    )[0, 1]

    correlation = np.corrcoef(
        error_sg[:, j],
        error_kf[:, j]
    )[0, 1]

    denominator = var_sg + var_kf - 2 * covariance

    if denominator <= 1e-12:
        alpha = 0.5
    else:
        alpha = (var_kf - covariance) / denominator

    # Constrain to convex combination
    alpha = np.clip(alpha, 0.0, 1.0)

    # Combined velocity estimate
    combined_velocity[:, j] = (
        alpha * sg_estimate[:, j]
        + (1 - alpha) * kf_estimate[:, j]
    )

    print(f"\nComponent: {component}")
    print(f"SG variance:       {var_sg:.6f}")
    print(f"Kalman variance:   {var_kf:.6f}")
    print(f"Error covariance:  {covariance:.6f}")
    print(f"Error correlation: {correlation:.6f}")
    print(f"Optimal SG weight: {alpha:.4f}")
    print(f"Kalman weight:     {1-alpha:.4f}")


# =====================================================
#  COMBINED ESTIMATOR METRICS
# =====================================================

print("\n========== COMBINED ESTIMATOR ==========")

for j, component in enumerate(components):

    result = calculate_metrics(
        velocity_reference[:, j],
        combined_velocity[:, j]
    )

    print(f"\n{component}")

    for name, value in result.items():
        print(f"{name}: {value:.6f}")


# Speed must be calculated from the combined vector.
combined_speed = np.linalg.norm(
    combined_velocity,
    axis=1
)

reference_speed = np.linalg.norm(
    velocity_reference,
    axis=1
)

speed_metrics = calculate_metrics(
    reference_speed,
    combined_speed
)

print("\nCombined speed metrics:")

for name, value in speed_metrics.items():
    print(f"{name}: {value:.6f}")


# =====================================================
#  PLOT: ERROR COMPENSATION
# =====================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for j, ax in enumerate(axes):

    ax.plot(
        t_eval,
        error_sg[:, j],
        label="SG error"
    )

    ax.plot(
        t_eval,
        error_kf[:, j],
        label="Kalman error"
    )

    combined_error = (
        combined_velocity[:, j]
        - velocity_reference[:, j]
    )

    ax.plot(
        t_eval,
        combined_error,
        label="Combined error",
        linewidth=2
    )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel(f"{components[j]} error (m/s)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("Error Compensation: SG + Kalman")

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "estimator_fusion_errors.png",
    dpi=200
)

plt.show()
plt.close(fig)


comparison_data = {
    "time_s": t_eval,

    "vx_true": velocity_reference[:, 0],
    "vy_true": velocity_reference[:, 1],
    "vz_true": velocity_reference[:, 2],

    "vx_sg11": sg11[:, 0],
    "vy_sg11": sg11[:, 1],
    "vz_sg11": sg11[:, 2],

    "vx_kalman": kalman_eval[:, 0],
    "vy_kalman": kalman_eval[:, 1],
    "vz_kalman": kalman_eval[:, 2],

    "speed_true": np.linalg.norm(
        velocity_reference, axis=1
    ),

    "speed_sg11": np.linalg.norm(
        sg11, axis=1
    ),

    "speed_kalman": np.linalg.norm(
        kalman_eval, axis=1
    )
}

comparison_df = pd.DataFrame(comparison_data)

comparison_df.to_csv(
    DATA_DIR / "sg11_kalman_comparison.csv",
    index=False
)


# =====================================================
# 11. PLOT: RMSE VS WINDOW SIZE
# =====================================================

quantities = ["vx", "vy", "vz", "speed"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, quantity in zip(axes.flat, quantities):

    sg_subset = metrics_df[
        (metrics_df["Method"] == "Savitzky-Golay") &
        (metrics_df["Quantity"] == quantity)
    ].sort_values("Window")

    kalman_value = metrics_df[
        (metrics_df["Method"] == "Kalman") &
        (metrics_df["Quantity"] == quantity)
    ]["RMSE"].iloc[0]

    ax.plot(
        sg_subset["Window"],
        sg_subset["RMSE"],
        marker="o",
        label="Savitzky-Golay"
    )

    ax.axhline(
        kalman_value,
        linestyle="--",
        label="Kalman"
    )

    ax.set_xticks(SG_WINDOWS)
    ax.set_xlabel("Window length")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title(f"RMSE of {quantity}")
    ax.grid(True)
    ax.legend()

fig.suptitle("RMSE vs Savitzky-Golay Window Length")

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "rmse_vs_window_size.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 12. PLOT: 3D VECTOR RMSE
# =====================================================

sg_vector = vector_df[
    vector_df["Method"] == "Savitzky-Golay"
]

plt.figure(figsize=(9, 5))

plt.plot(
    sg_vector["Window"],
    sg_vector["Vector_RMSE"],
    marker="o",
    label="Savitzky-Golay"
)

plt.axhline(
    kalman_vector_rmse,
    linestyle="--",
    label="Kalman"
)

plt.xticks(SG_WINDOWS)
plt.xlabel("SG window length")
plt.ylabel("3D velocity vector RMSE (m/s)")
plt.title("Overall Velocity Estimation Accuracy")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "vector_rmse_vs_window.png",
    dpi=200
)

plt.close()


# =====================================================
# 13. PLOT: SG 11 VS KALMAN
# =====================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for j, ax in enumerate(axes):

    ax.plot(
        t_eval,
        velocity_reference[:, j],
        label="Analytical reference",
        linewidth=2
    )

    ax.plot(
        t_eval,
        sg11[:, j],
        label="SG (window 11)"
    )

    ax.plot(
        t_eval,
        kalman_eval[:, j],
        label="Kalman"
    )

    ax.set_ylabel(["vx", "vy", "vz"][j] + " (m/s)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("SG Window 11 vs Kalman")

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "sg11_vs_kalman.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 14. PLOT: ERROR COMPARISON
# =====================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 11))

sg_error = sg11 - velocity_reference
kf_error = kalman_eval - velocity_reference

for j, ax in enumerate(axes[:3]):

    ax.plot(
        t_eval,
        sg_error[:, j],
        label="SG error"
    )

    ax.plot(
        t_eval,
        kf_error[:, j],
        label="Kalman error"
    )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel(
        ["vx", "vy", "vz"][j] + " error (m/s)"
    )

    ax.grid(True)
    ax.legend()


sg_speed_error = (
    np.linalg.norm(sg11, axis=1) - true_speed
)

kf_speed_error = (
    np.linalg.norm(kalman_eval, axis=1) - true_speed
)

axes[3].plot(
    t_eval,
    sg_speed_error,
    label="SG speed error"
)

axes[3].plot(
    t_eval,
    kf_speed_error,
    label="Kalman speed error"
)

axes[3].axhline(0, linestyle="--")
axes[3].set_ylabel("Speed error (m/s)")
axes[3].set_xlabel("Time (s)")
axes[3].grid(True)
axes[3].legend()

fig.suptitle("SG Window 11 vs Kalman: Error")

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "sg11_kalman_errors.png",
    dpi=200
)

plt.close(fig)


# =====================================================
# 14.5. SEPARATE GRAPHS FOR EVERY SG WINDOW vs KALMAN
# =====================================================

for window in SG_WINDOWS:

    print(f"\nGenerating plots for SG window = {window}")

    # SG estimate at common evaluation timestamps
    sg_estimate = sg_results[window][indices]

    # Speed magnitude
    sg_speed = np.linalg.norm(
        sg_estimate, axis=1
    )

    # Errors
    sg_error = sg_estimate - velocity_reference
    kalman_error = kalman_eval - velocity_reference

    sg_speed_error = sg_speed - true_speed
    kalman_speed_error = (
        np.linalg.norm(kalman_eval, axis=1) - true_speed
    )

    prefix = f"window_{window:02d}"

    # =================================================
    # GRAPH 1: VELOCITY COMPONENTS
    # =================================================

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    component_names = ["vx", "vy", "vz"]

    for j, ax in enumerate(axes):

        ax.plot(
            t_eval,
            velocity_reference[:, j],
            label="Analytical reference",
            linewidth=2
        )

        ax.plot(
            t_eval,
            sg_estimate[:, j],
            label=f"Savitzky-Golay (window={window})"
        )

        ax.plot(
            t_eval,
            kalman_eval[:, j],
            label="Kalman Filter"
        )

        ax.set_ylabel(f"{component_names[j]} (m/s)")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        f"Velocity Components: SG Window {window} vs Kalman"
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / f"{prefix}_velocity_comparison.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # =================================================
    # GRAPH 2: SPEED MAGNITUDE
    # =================================================

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        t_eval,
        true_speed,
        label="Analytical speed",
        linewidth=2
    )

    ax.plot(
        t_eval,
        sg_speed,
        label=f"Savitzky-Golay (window={window})"
    )

    ax.plot(
        t_eval,
        np.linalg.norm(kalman_eval, axis=1),
        label="Kalman Filter"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")

    ax.set_title(
        f"Speed Magnitude: SG Window {window} vs Kalman"
    )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / f"{prefix}_speed_comparison.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # =================================================
    # GRAPH 3: VELOCITY ERRORS
    # =================================================

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    for j, ax in enumerate(axes[:3]):

        ax.plot(
            t_eval,
            sg_error[:, j],
            label=f"SG window {window} error"
        )

        ax.plot(
            t_eval,
            kalman_error[:, j],
            label="Kalman error"
        )

        ax.axhline(0, linestyle="--")

        ax.set_ylabel(
            f"{component_names[j]} error (m/s)"
        )

        ax.grid(True)
        ax.legend()

    axes[3].plot(
        t_eval,
        sg_speed_error,
        label=f"SG window {window} speed error"
    )

    axes[3].plot(
        t_eval,
        kalman_speed_error,
        label="Kalman speed error"
    )

    axes[3].axhline(0, linestyle="--")

    axes[3].set_ylabel("Speed error (m/s)")
    axes[3].set_xlabel("Time (s)")

    axes[3].grid(True)
    axes[3].legend()

    fig.suptitle(
        f"Estimation Errors: SG Window {window} vs Kalman"
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / f"{prefix}_error_comparison.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)


print("\nAll individual SG vs Kalman plots generated.")

# =====================================================
# 15. FINISH
# =====================================================

print("\n========== EXPERIMENT COMPLETED ==========")

print(f"Data directory: {DATA_DIR}")
print(f"Plots directory: {PLOTS_DIR}")

print("\nAll measurements are SYNTHETIC.")
