import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import differential_evolution

# =========================================================
# Autonomous Vehicle Lateral Model Parameters
# =========================================================
L = 0.15       # Wheelbase (15 cm for 21cm chassis)
v = 0.50       # Constant forward speed in m/s (from simulator)
tau = 0.05     # Steering actuator time constant (delay)

# Vehicle Transfer Function: G(s) = (v^2 / L) / [s^2 * (tau*s + 1)]
# Denominator: tau * s^3 + s^2 + 0*s + 0
gain = (v**2) / L
numerator = [gain]
denominator = [tau, 1.0, 0.0, 0.0]

vehicle = ctrl.TransferFunction(numerator, denominator)

# =========================================================
# Simulation Time (Tracking a step lane-change of 1 meter)
# =========================================================
time = np.linspace(0, 5, 1000)

# Initial baseline gains (from lane_detection_node)
Kp0 = 1.0
Ki0 = 0.0      # Ki should remain near zero for lane tracking
Kd0 = 0.5

print("=" * 55)
print("Autonomous Vehicle PID Optimization (Lateral Control)")
print("=" * 55)
print("\nVehicle Lateral Transfer Function:\n", vehicle)

# PID Controller Definition
def create_pid(Kp, Ki, Kd):
    # To make the derivative physically realizable, add a low-pass filter
    # C(s) = Kp + Ki/s + Kd*s / (0.01*s + 1)
    # Standard ideal form: (Kd*s^2 + Kp*s + Ki) / s
    numerator = [Kd, Kp, Ki]
    denominator = [1, 0]
    return ctrl.TransferFunction(numerator, denominator)

# Closed Loop Simulation
def simulate_system(Kp, Ki, Kd):
    pid = create_pid(Kp, Ki, Kd)
    closed_loop = ctrl.feedback(pid * vehicle, 1)
    t_out, y_out = ctrl.step_response(closed_loop, T=time)
    return t_out, y_out

# Cost Function (ISE with Actuator Penalty)
def objective(params):
    Kp, Ki, Kd = params
    try:
        t_out, y = simulate_system(Kp, Ki, Kd)
        error = 1.0 - y  # Step target = 1 unit offset
        
        # Check stability: if output explodes or has NaN
        if np.any(np.isnan(y)) or np.max(np.abs(y)) > 10.0:
            return 1e9

        ise = np.trapezoid(error**2, t_out)
        
        # Penalize excessive gains to keep steering smooth and avoid oscillation
        penalty = 0.001 * (Kp**2 + Kd**2) + 0.1 * (Ki**2)
        return ise + penalty
    except Exception:
        return 1e9

# Search Space (Bounded for stable small-scale steering)
bounds = [
    (0.1, 5.0),   # Kp
    (0.0, 0.5),   # Ki (tightly bounded to prevent integral windup)
    (0.05, 3.0)   # Kd
]

# PID Optimization using Differential Evolution
print("\nOptimizing PID Parameters via Differential Evolution ...")
result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=40,
    popsize=15,
    tol=1e-4,
    polish=True
)

Kp_opt, Ki_opt, Kd_opt = result.x

print("\nOptimization Finished Successfully.\n")
print(f"Optimal Kp = {Kp_opt:.4f}")
print(f"Optimal Ki = {Ki_opt:.4f}")
print(f"Optimal Kd = {Kd_opt:.4f}")
print(f"Minimum Cost = {result.fun:.6f}")

# Simulation comparison
t_before, y_before = simulate_system(Kp0, Ki0, Kd0)
t_after, y_after = simulate_system(Kp_opt, Ki_opt, Kd_opt)

# Plot Results
plt.figure(figsize=(9, 5))
plt.plot(t_before, y_before, 'r--', linewidth=2, label=f'Initial: Kp={Kp0}, Kd={Kd0}')
plt.plot(t_after, y_after, 'b', linewidth=2, label=f'Optimized: Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}')
plt.axhline(1.0, color='g', linestyle=':', label='Target Centerline')
plt.title("Vehicle Lateral Step Response (Lane Tracking)")
plt.xlabel("Time (s)")
plt.ylabel("Lateral Position Offset (e_y)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()