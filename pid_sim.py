import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class PID:
    """
    A simple PID controller class, similar to what you would implement
    on an Arduino.
    """
    def __init__(self, kp, ki, kd, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None

    def compute(self, setpoint, measured_value, current_time):
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0
        
        dt = current_time - self.prev_time
        if dt <= 0: return 0.0 # Prevent division by zero or negative time

        error = setpoint - measured_value
        
        # Proportional term
        P = self.kp * error
        
        # Integral term
        self.integral += error * dt
        # Clamp integral to prevent windup (optional but recommended)
        self.integral = max(min(self.integral, self.max_out), self.min_out)
        I = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative
        
        # Calculate total output
        output = P + I + D
        
        # Clamp output
        output = max(min(output, self.max_out), self.min_out)
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output

class Drone:
    """
    Simulates a quadcopter's physical dynamics.
    """
    def __init__(self, mass=1.0, arm_length=0.25, Ixx=0.01, Iyy=0.01, Izz=0.02):
        # Physical constants
        self.mass = mass  # kg
        self.gravity = 9.81 # m/s^2
        self.arm_length = arm_length # meters
        self.Ixx = Ixx # Inertia x
        self.Iyy = Iyy # Inertia y
        self.Izz = Izz # Inertia z
        
        # State: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        # Position (x,y,z), Euler Angles (phi, theta, psi)
        # Linear Velocities (u,v,w), Angular Velocities (p,q,r)
        self.state = np.zeros(12)
        self.state[2] = 0.0 # Start on the ground
        
    def dynamics(self, state, motor_inputs):
        """
        Calculates the derivatives of the state vector based on motor inputs.
        motor_inputs: [f1, f2, f3, f4] (Forces from 4 motors)
        """
        # Unpack state
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        
        f1, f2, f3, f4 = motor_inputs
        
        # Total Thrust
        F_total = f1 + f2 + f3 + f4
        
        # Torques (Assuming + config: 1(front), 2(right), 3(back), 4(left))
        # Note: This mixing varies by frame type (X vs +). 
        # Using a standard X-configuration for this example:
        # 1(front-right, CCW), 2(back-left, CCW), 3(front-left, CW), 4(back-right, CW)
        
        # Simplify for simulation:
        # Roll torque (diff between left and right)
        tau_phi = self.arm_length * ((f2 + f3) - (f1 + f4)) # Just an example mixing
        # Pitch torque (diff between front and back)
        tau_theta = self.arm_length * ((f1 + f3) - (f2 + f4))
        # Yaw torque (diff between CW and CCW props)
        # Assuming torque is proportional to thrust with const C_yaw
        C_yaw = 0.02 
        tau_psi = C_yaw * ((f1 + f2) - (f3 + f4))
        
        # Rotation Matrix (Body to Earth)
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        # Linear Acceleration (Newton's 2nd Law)
        # Acceleration in Earth Frame
        ax = (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)) * F_total / self.mass
        ay = (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)) * F_total / self.mass
        az = -self.gravity + (np.cos(phi)*np.cos(theta)) * F_total / self.mass
        
        # Angular Acceleration (Euler's equations for rigid body dynamics)
        p_dot = (tau_phi - (self.Izz - self.Iyy)*q*r) / self.Ixx
        q_dot = (tau_theta - (self.Ixx - self.Izz)*p*r) / self.Iyy
        r_dot = (tau_psi - (self.Iyy - self.Ixx)*p*q) / self.Izz
        
        # Return state derivatives
        return np.array([u, v, w, p, q, r, ax, ay, az, p_dot, q_dot, r_dot])

    def update(self, motor_inputs, dt):
        """
        Simple Euler integration to advance simulation.
        """
        derivatives = self.dynamics(self.state, motor_inputs)
        self.state += derivatives * dt
        
        # Hard floor constraint (cannot go below z=0)
        if self.state[2] < 0:
            self.state[2] = 0
            self.state[8] = 0 # zero z-velocity

# --- Simulation Setup ---

def run_simulation():
    # --- USER CONFIGURATION: ENTER YOUR DRONE SPECS HERE ---
    drone_mass = 1.2        # kg
    arm_length = 0.22       # meters (center to motor)
    Ixx = 0.015             # kg*m^2
    Iyy = 0.015             # kg*m^2
    Izz = 0.025             # kg*m^2
    
    drone = Drone(mass=drone_mass, arm_length=arm_length, Ixx=Ixx, Iyy=Iyy, Izz=Izz)
    
    # Simulation Parameters
    dt = 0.02 # 50Hz update rate
    sim_time = 10.0 # seconds
    steps = int(sim_time / dt)
    times = np.linspace(0, sim_time, steps)
    
    # PID Controllers
    # Tuning these Kp, Ki, Kd values is the goal!
    # Altitude PID
    pid_alt = PID(kp=12.0, ki=1.0, kd=4.0, output_limits=(0, 25)) 
    
    # Roll/Pitch PIDs (Stabilization)
    pid_roll = PID(kp=2.5, ki=0.0, kd=0.8, output_limits=(-3, 3))
    pid_pitch = PID(kp=2.5, ki=0.0, kd=0.8, output_limits=(-3, 3))
    
    # Target Setpoints
    target_alt = 5.0 # Fly to 5 meters
    target_roll = 0.0 # Stay level
    target_pitch = 0.0 # Stay level
    
    # Data logging
    history = {'x': [], 'y': [], 'z': [], 'phi': [], 'theta': []}
    
    print(f"Starting Simulation ({sim_time}s)...")
    
    for t in times:
        # 1. Get current state
        z = drone.state[2]
        phi = drone.state[3]   # Roll
        theta = drone.state[4] # Pitch
        
        # 2. Compute PID outputs
        # Altitude control adds to base throttle
        throttle_adj = pid_alt.compute(target_alt, z, t)
        base_thrust = (drone.mass * drone.gravity) / 4.0 # Hover thrust per motor
        
        # Attitude control calculates necessary torque (differential thrust)
        roll_correction = pid_roll.compute(target_roll, phi, t)
        pitch_correction = pid_pitch.compute(target_pitch, theta, t)
        
        # 3. Mix outputs to motors
        # Simple mixing logic (conceptual):
        # Motor = Base + AltitudePID + AttitudePID
        m1 = base_thrust + throttle_adj + pitch_correction - roll_correction
        m2 = base_thrust + throttle_adj - pitch_correction + roll_correction
        m3 = base_thrust + throttle_adj + pitch_correction + roll_correction
        m4 = base_thrust + throttle_adj - pitch_correction - roll_correction
        
        # Clamp motor values to realistic positive limits
        motor_inputs = np.clip([m1, m2, m3, m4], 0, 50)
        
        # 4. Update Physics
        drone.update(motor_inputs, dt)
        
        # Log data
        history['x'].append(drone.state[0])
        history['y'].append(drone.state[1])
        history['z'].append(drone.state[2])
        history['phi'].append(drone.state[3])
        history['theta'].append(drone.state[4])

    print("Simulation Complete.")
    
    # --- Plotting ---
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(history['x'], history['y'], history['z'], label='Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('Drone 3D Flight Path')
    ax1.legend()
    
    # Altitude vs Time
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(times, history['z'], 'b-', label='Current Altitude')
    ax2.plot(times, [target_alt]*len(times), 'r--', label='Target Altitude')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Control Response')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()