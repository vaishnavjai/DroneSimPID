"""
Quadcopter PID Flight Simulation
Simulates altitude and attitude control using cascaded PID loops. 
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import NamedTuple


# ============================================================================
# CONFIGURATION (Dataclasses for clean, typed configuration)
# ============================================================================

@dataclass
class PIDGains: 
    """PID controller gain parameters."""
    kp:  float = 0.0      # Proportional gain
    ki: float = 0.0      # Integral gain
    kd:  float = 0.0      # Derivative gain
    limit:  float = 10.0  # Output saturation limit


@dataclass
class DroneParams:
    """Physical properties of the quadcopter."""
    mass: float = 1.4           # kg
    arm_length: float = 0.22    # meters (motor to center)
    Ixx: float = 0.015          # Roll moment of inertia (kg·m²)
    Iyy:  float = 0.015          # Pitch moment of inertia (kg·m²)
    Izz: float = 0.025          # Yaw moment of inertia (kg·m²)
    torque_coeff: float = 0.02  # Thrust-to-torque ratio for yaw
    motor_max: float = 100.0    # Maximum thrust per motor (N)
    gravity: float = 9.81       # m/s²


@dataclass
class SimConfig:
    """Simulation parameters."""
    dt: float = 0.02         # Timestep (s) → 50Hz control loop
    duration: float = 5.0    # Total simulation time (s)
    target_altitude: float = 5.0   # meters
    target_roll: float = 0.0       # radians (level flight)
    target_pitch: float = 0.0      # radians
    target_yaw: float = 0.0        # radians


class DroneState(NamedTuple):
    """Drone state vector for clarity."""
    z: float = 0.0          # Altitude (m)
    phi: float = 0.0        # Roll angle (rad)
    theta: float = 0.0      # Pitch angle (rad)
    psi: float = 0.0        # Yaw angle (rad)
    z_dot: float = 0.0      # Vertical velocity (m/s)
    phi_dot: float = 0.0    # Roll rate (rad/s)
    theta_dot: float = 0.0  # Pitch rate (rad/s)
    psi_dot: float = 0.0    # Yaw rate (rad/s)


# ============================================================================
# PID CONTROLLER
# ============================================================================

class PIDController:
    """
    Discrete PID controller with anti-windup. 
    
    Features:
        - Integral clamping to prevent windup
        - Output saturation
        - Derivative filtering (optional, not shown for simplicity)
    """
    
    def __init__(self, gains: PIDGains):
        self.gains = gains
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measured:  float, dt: float) -> float:
        """
        Calculate PID output. 
        
        Args:
            setpoint:  Desired target value
            measured:  Current sensor reading
            dt:  Time delta since last call (s)
        
        Returns: 
            Control signal (saturated to ±limit)
        """
        error = setpoint - measured
        
        # Proportional term
        p_term = self. gains.kp * error
        
        # Integral term with anti-windup clamping
        self._integral += error * dt
        self._integral = np. clip(
            self._integral, 
            -self. gains.limit / max(self.gains.ki, 1e-6),  # Prevent div-by-zero
            self.gains.limit / max(self.gains.ki, 1e-6)
        )
        i_term = self. gains.ki * self._integral
        
        # Derivative term (on error; consider derivative-on-measurement for real systems)
        derivative = (error - self._prev_error) / dt
        d_term = self. gains.kd * derivative
        self._prev_error = error
        
        # Sum and saturate output
        output = p_term + i_term + d_term
        return np. clip(output, -self.gains.limit, self.gains.limit)

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0


# ============================================================================
# QUADCOPTER DYNAMICS
# ============================================================================

class Quadcopter:
    """
    Simplified quadcopter dynamics model. 
    
    Motor Layout (X-configuration, top view):
    
            Front
        M3 (CCW)   M1 (CW)
              \\ /
               X
              / \\
        M2 (CW)   M4 (CCW)
            Back
    
    - CW/CCW indicates propeller spin direction
    - Positive roll (phi): right side down
    - Positive pitch (theta): nose down  
    - Positive yaw (psi): clockwise from above
    """
    
    def __init__(self, params: DroneParams):
        self.params = params
        self._state = np.zeros(8)  # [z, phi, theta, psi, z_dot, phi_dot, theta_dot, psi_dot]

    @property
    def state(self) -> DroneState:
        """Return current state as named tuple for readability."""
        return DroneState(*self._state)

    def step(self, motor_thrusts: np. ndarray, dt: float) -> None:
        """
        Advance physics simulation by one timestep.
        
        Args:
            motor_thrusts:  Array [M1, M2, M3, M4] thrust values (N)
            dt:  Timestep (s)
        """
        m1, m2, m3, m4 = motor_thrusts
        p = self.params
        
        # --- Force and Torque Calculations ---
        # Total vertical thrust
        thrust_total = m1 + m2 + m3 + m4
        
        # Roll torque:  (Right motors) - (Left motors)
        # Positive torque → positive phi_dot → right side down
        tau_roll = p.arm_length * ((m1 + m4) - (m2 + m3))
        
        # Pitch torque: (Front motors) - (Back motors)
        # Positive torque → positive theta_dot → nose down
        tau_pitch = p.arm_length * ((m1 + m3) - (m2 + m4))
        
        # Yaw torque: (CW props) - (CCW props)
        # Due to reaction torque from propeller spin
        tau_yaw = p.torque_coeff * ((m1 + m2) - (m3 + m4))
        
        # --- Accelerations (simplified, decoupled model) ---
        z_acc = (thrust_total / p.mass) - p.gravity
        phi_acc = tau_roll / p. Ixx
        theta_acc = tau_pitch / p.Iyy
        psi_acc = tau_yaw / p. Izz
        
        # --- Euler Integration ---
        # Update velocities
        self._state[4] += z_acc * dt      # z_dot
        self._state[5] += phi_acc * dt    # phi_dot
        self._state[6] += theta_acc * dt  # theta_dot
        self._state[7] += psi_acc * dt    # psi_dot
        
        # Update positions
        self._state[0] += self._state[4] * dt  # z
        self._state[1] += self._state[5] * dt  # phi
        self._state[2] += self._state[6] * dt  # theta
        self._state[3] += self._state[7] * dt  # psi
        
        # Ground collision constraint
        if self._state[0] < 0:
            self._state[0] = 0.0
            self._state[4] = 0.0

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = np.zeros(8)


# ============================================================================
# MOTOR MIXER
# ============================================================================

def mix_motors(
    throttle: float,
    roll_cmd: float,
    pitch_cmd:  float,
    yaw_cmd: float,
    motor_max: float
) -> np.ndarray:
    """
    Convert control commands to individual motor thrusts.
    
    Standard X-frame mixing matrix:
        M1 = throttle + roll - pitch + yaw  (Front-Right, CW)
        M2 = throttle - roll + pitch + yaw  (Back-Left, CW)
        M3 = throttle - roll - pitch - yaw  (Front-Left, CCW)
        M4 = throttle + roll + pitch - yaw  (Back-Right, CCW)
    
    Args:
        throttle: Base thrust command (N per motor)
        roll_cmd: Roll correction (+ = right down)
        pitch_cmd: Pitch correction (+ = nose down)
        yaw_cmd: Yaw correction (+ = clockwise)
        motor_max: Maximum thrust per motor
    
    Returns: 
        Array of 4 motor thrust values [M1, M2, M3, M4]
    """
    motors = np.array([
        throttle + roll_cmd - pitch_cmd + yaw_cmd,  # M1: Front-Right
        throttle - roll_cmd + pitch_cmd + yaw_cmd,  # M2: Back-Left
        throttle - roll_cmd - pitch_cmd - yaw_cmd,  # M3: Front-Left
        throttle + roll_cmd + pitch_cmd - yaw_cmd,  # M4: Back-Right
    ])
    return np.clip(motors, 0, motor_max)


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(
    drone_params: DroneParams,
    sim_config: SimConfig,
    altitude_gains: PIDGains,
    roll_gains: PIDGains,
    pitch_gains: PIDGains,
    yaw_gains: PIDGains
) -> dict:
    """
    Execute the flight simulation.
    
    Returns:
        Dictionary containing time series data for plotting. 
    """
    # Initialize systems
    drone = Quadcopter(drone_params)
    pid_altitude = PIDController(altitude_gains)
    pid_roll = PIDController(roll_gains)
    pid_pitch = PIDController(pitch_gains)
    pid_yaw = PIDController(yaw_gains)
    
    # Time vector
    time_points = np.arange(0, sim_config.duration, sim_config.dt)
    
    # Data logging
    history = {
        'time':  time_points,
        'altitude': [],
        'roll': [],
        'pitch': [],
        'yaw':  []
    }
    
    # Hover thrust per motor (to counteract gravity)
    hover_thrust = (drone_params.mass * drone_params.gravity) / 4.0
    
    print(f"Starting simulation:  {sim_config.duration}s at {1/sim_config. dt:.0f}Hz")
    
    for t in time_points:
        state = drone.state
        
        # --- PID Control Loop ---
        throttle_adj = pid_altitude.compute(
            sim_config.target_altitude, state. z, sim_config.dt
        )
        roll_cmd = pid_roll. compute(
            sim_config. target_roll, state.phi, sim_config.dt
        )
        pitch_cmd = pid_pitch.compute(
            sim_config.target_pitch, state. theta, sim_config.dt
        )
        yaw_cmd = pid_yaw.compute(
            sim_config. target_yaw, state.psi, sim_config.dt
        )
        
        # --- Motor Mixing ---
        base_throttle = hover_thrust + throttle_adj
        motor_thrusts = mix_motors(
            base_throttle, roll_cmd, pitch_cmd, yaw_cmd, drone_params.motor_max
        )
        
        # --- Update Physics ---
        drone.step(motor_thrusts, sim_config.dt)
        
        # --- Log Data ---
        history['altitude'].append(state.z)
        history['roll'].append(state.phi)
        history['pitch'].append(state. theta)
        history['yaw'].append(state.psi)
    
    print("Simulation complete.")
    return history


def plot_results(history:  dict, sim_config: SimConfig) -> None:
    """Generate plots of simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Quadcopter PID Control Response', fontsize=14, fontweight='bold')
    
    plots = [
        ('altitude', sim_config.target_altitude, 'Altitude (m)', 'tab: blue'),
        ('roll', sim_config.target_roll, 'Roll (rad)', 'tab:orange'),
        ('pitch', sim_config. target_pitch, 'Pitch (rad)', 'tab:green'),
        ('yaw', sim_config.target_yaw, 'Yaw (rad)', 'tab:red'),
    ]
    
    for ax, (key, target, ylabel, color) in zip(axes.flat, plots):
        ax.plot(history['time'], history[key], color=color, linewidth=1. 5, label='Actual')
        ax.axhline(target, color='gray', linestyle='--', linewidth=1, label='Target')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{key. capitalize()} Response')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # --- Configure Simulation ---
    drone_params = DroneParams(
        mass=1.4,
        arm_length=0.22,
        Ixx=0.015,
        Iyy=0.015,
        Izz=0.025,
    )
    
    sim_config = SimConfig(
        dt=0.02,
        duration=5.0,
        target_altitude=5.0,
        target_roll=0.0,
        target_pitch=0.0,
        target_yaw=0.0,
    )
    
    # PID Gains (tune these for your drone)
    altitude_gains = PIDGains(kp=15.0, ki=2.0, kd=8.0, limit=50.0)
    roll_gains = PIDGains(kp=3.0, ki=0.0, kd=1.0, limit=5.0)
    pitch_gains = PIDGains(kp=3.0, ki=0.0, kd=1.0, limit=5.0)
    yaw_gains = PIDGains(kp=2.0, ki=0.0, kd=0.5, limit=3.0)
    
    # --- Run ---
    results = run_simulation(
        drone_params, sim_config,
        altitude_gains, roll_gains, pitch_gains, yaw_gains
    )
    
    plot_results(results, sim_config)