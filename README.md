# DroneSimPID
Libraries: numpy for math, matplotlib for plotting graph, os for saving graph as png

# Configuration:
## Drone physical parameters
MASS = 1.4          # kg (The weight of the drone)
GRAVITY = 9.81      # m/s² (Earth's gravity)

## Simulation settings
DT = 0.02           # The "refresh rate" of the simulation. 0.02s = 50 times per second.
DURATION = 10.0     # Total flight time to simulate (10 seconds).
TARGET_ALT = 2.0    # The drone wants to reach 2 meters height.

These are values required for simulation as these are the values we use in the real world. 

# PID Values
KP = 32.4   # Proportional Gain
KI = 0      # Integral Gain
KD = 16.0   # Derivative Gain
LIMIT = 50.0 # Max force the PID can add/subtract 
**These were the values after tuning. Graph has been sent to the channel**

# PID Logic
-> Kp (Proportional gain term) is the control system's immediate reaction to current error. Lasrger the error, larger the p value 
-> Ki (Integral gain term) is  the system's reaction to accumalated error (hence the name, it integrates the graph and considers error over time).
-> Kd (Differential gain term) is the system's way of predicting future error by calculating slope of the error via derivatives
**These have been calculated using NumPy, as they need Differential and integral calculus that the standard python or pymath library do not provide.**

# Class: PIDController
   
    Initialization
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.integral = 0.0      # Stores the accumulated error over time
        self.prev_error = 0.0    # Stores the error from the last frame

# The Simulation Loop
**50Hz sampling time*
he PID only needs to handle the *difference* or corrections, not the whole weight.
Initial State: Drone starts at altitude 0 (ground) with 0 velocity.
Step 1. Ask PID how much to adjust thrust based on current altitude (z)
Step 2. Physics (Newton's Second Law): F = ma  ->  a = F / m
Step 3. Update Velocity (z_dot) and Position (z) with the updated values via Iterated loop

# Metrics Calculation
-> Overshoot
-> Settling time
-> Max altitude
-> Steady state error
-> Rise time
We calculate all these values with **compute_metrics(time, altitude, target=TARGET_ALT)*

# Plotting via MatPlotLib
fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, altitude, ...)      # Draws the blue line (Drone path)
    ax.axhline(TARGET_ALT, ...)       # Draws the red dashed line (Target)
    ax.axhline(TARGET_ALT * 1.02, ...) # Draws the grey "Good Zone" lines
    
-> SAVING
    os.makedirs('results', exist_ok=True) # Creates 'results' folder if it doesn't exist
    fig.savefig('results/hover_response.png', ...) # Saves the image