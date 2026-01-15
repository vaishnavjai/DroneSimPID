"""
Hover PID Tuner - Simple altitude hold simulation
Finds and validates PID values for stable hovering.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

# Drone physical parameters
MASS = 1.4          # kg
GRAVITY = 9.81      # m/s²

# Simulation settings
DT = 0.02           # 50Hz control loop
DURATION = 10.0     # seconds
TARGET_ALT = 2.0    # meters

# PID Gains for altitude (TUNE THESE)
KP = 32.4   # Proportional: how aggressively to correct error
KI = 0 # Integral: eliminate steady-state error
KD = 16.0    # Derivative: dampen oscillations
LIMIT = 50.0  # Max thrust adjustment (N)


# ============================================================================
# PID CONTROLLER
# ============================================================================

class PIDController:
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measured, dt):
        error = setpoint - measured
        
        # P term
        p = self.kp * error
        
        # I term with anti-windup
        self.integral += error * dt
        max_i = self.limit / max(self.ki, 1e-6)
        self.integral = np.clip(self.integral, -max_i, max_i)
        i = self.ki * self.integral
        
        # D term
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        return np.clip(p + i + d, -self.limit, self.limit)


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_hover(kp, ki, kd, limit=LIMIT):
    """Run hover simulation and return metrics."""
    pid = PIDController(kp, ki, kd, limit)
    hover_thrust = MASS * GRAVITY  # Thrust needed to hover
    
    # State: [altitude, velocity]
    z, z_dot = 0.0, 0.0
    
    time = np.arange(0, DURATION, DT)
    altitude = []
    
    for _ in time:
        # PID computes thrust adjustment
        thrust_adj = pid.compute(TARGET_ALT, z, DT)
        total_thrust = hover_thrust + thrust_adj
        
        # Physics: F = ma → a = (F - mg) / m
        acc = (total_thrust / MASS) - GRAVITY
        z_dot += acc * DT
        z += z_dot * DT
        z = max(0, z)  # Ground constraint
        
        altitude.append(z)
    
    return time, np.array(altitude)


def compute_metrics(time, altitude, target=TARGET_ALT):
    """Compute performance metrics for hover response."""
    # Overshoot
    max_alt = np.max(altitude)
    overshoot = ((max_alt - target) / target) * 100 if max_alt > target else 0
    
    # Settling time (within 2% of target)
    tolerance = 0.02 * target
    settled_mask = np.abs(altitude - target) < tolerance
    if np.any(settled_mask):
        settling_idx = np.argmax(settled_mask)
        settling_time = time[settling_idx]
    else:
        settling_time = float('inf')
    
    # Steady-state error (last 1 second average)
    ss_samples = int(1.0 / DT)
    ss_error = abs(target - np.mean(altitude[-ss_samples:]))
    
    # Rise time (10% to 90% of target)
    try:
        t10 = time[np.argmax(altitude >= 0.1 * target)]
        t90 = time[np.argmax(altitude >= 0.9 * target)]
        rise_time = t90 - t10
    except:
        rise_time = float('inf')
    
    return {
        'overshoot': overshoot,
        'settling_time': settling_time,
        'steady_state_error': ss_error,
        'rise_time': rise_time
    }


def main():
    print("=" * 50)
    print("HOVER PID TUNER")
    print("=" * 50)
    print(f"\nDrone: {MASS}kg | Target: {TARGET_ALT}m | Duration: {DURATION}s")
    print(f"\nTesting PID: Kp={KP}, Ki={KI}, Kd={KD}")
    print("-" * 50)
    
    # Run simulation
    time, altitude = simulate_hover(KP, KI, KD, LIMIT)
    metrics = compute_metrics(time, altitude)
    
    # Print results
    print(f"\nRESULTS:")
    print(f"  Rise Time:        {metrics['rise_time']:.2f}s")
    print(f"  Overshoot:        {metrics['overshoot']:.1f}%")
    print(f"  Settling Time:    {metrics['settling_time']:.2f}s")
    print(f"  Steady-State Err: {metrics['steady_state_error']:.3f}m")
    
    # Evaluate quality
    print(f"\nEVALUATION:")
    good = True
    if metrics['overshoot'] > 10:
        print("  ⚠ Overshoot too high → Increase Kd or decrease Kp")
        good = False
    if metrics['settling_time'] > 3:
        print("  ⚠ Slow settling → Increase Kp or Ki")
        good = False
    if metrics['steady_state_error'] > 0.1:
        print("  ⚠ Steady-state error → Increase Ki")
        good = False
    if metrics['rise_time'] > 2:
        print("  ⚠ Slow rise time → Increase Kp")
        good = False
    if good:
        print("  ✓ Good hover response!")
    
    # Print final PID values
    print(f"\n{'=' * 50}")
    print("RECOMMENDED PID VALUES FOR HOVER:")
    print(f"  Kp = {KP}")
    print(f"  Ki = {KI}")
    print(f"  Kd = {KD}")
    print(f"  Output Limit = {LIMIT}")
    print("=" * 50)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, altitude, 'b-', linewidth=2, label='Altitude')
    ax.axhline(TARGET_ALT, color='r', linestyle='--', label=f'Target ({TARGET_ALT}m)')
    ax.axhline(TARGET_ALT * 1.02, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(TARGET_ALT * 0.98, color='gray', linestyle=':', alpha=0.5, label='±2% Band')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Hover Response | Kp={KP}, Ki={KI}, Kd={KD}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, DURATION)
    ax.set_ylim(0, TARGET_ALT * 1.5)
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/hover_response.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: results/hover_response.png")
    
    if os.environ.get('DISPLAY'):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
