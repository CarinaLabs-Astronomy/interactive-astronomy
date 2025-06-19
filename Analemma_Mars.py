import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Constants and Parameters for MARS ---
NUM_FRAMES = 687  # One frame for each Martian sol in its year
# Martian Orbital Parameters
MARS_YEAR_IN_EARTH_DAYS = 687.0
MARS_OBLIQUITY_DEG = 25.19  # Axial Tilt
MARS_OBLIQUITY_RAD = np.deg2rad(MARS_OBLIQUITY_DEG)
MARS_ECCENTRICITY = 0.0934

# --- 2. Pre-calculate the Analemma Path ---
# We'll use a simplified physical model to calculate the two components of the analemma.

# 'sols' is our time variable, from 0 to 686
sols = np.arange(NUM_FRAMES)

# a) Mean Anomaly: The angle of a "mean" Mars in a perfect circular orbit
mean_anomaly = 2 * np.pi * sols / MARS_YEAR_IN_EARTH_DAYS

# b) True Anomaly: The actual angle in an elliptical orbit.
# A good approximation that captures the main effect of eccentricity.
true_anomaly = mean_anomaly + 2 * MARS_ECCENTRICITY * np.sin(mean_anomaly)

# c) Declination (Vertical Component): This is due to the axial tilt.
# It's a sine wave whose argument is the planet's position in its orbit (true anomaly).
declination_rad = np.arcsin(np.sin(MARS_OBLIQUITY_RAD) * np.sin(true_anomaly))
declination_deg = np.rad2deg(declination_rad)

# d) Equation of Time (Horizontal Component): This is the difference between mean and true time.
# Convert the angular difference to minutes of time. 1 day = 1440 min.
equation_of_time_min = (4 * 1440 / (2*np.pi)) * (mean_anomaly - true_anomaly)

# --- 3. Find Key Points for Labeling ---
summer_solstice_idx = np.argmax(declination_deg)
winter_solstice_idx = np.argmin(declination_deg)
# Find aphelion (slowest orbital speed, narrowest part of analemma)
aphelion_idx = np.argmin(true_anomaly - mean_anomaly)
# Find perihelion (fastest orbital speed, widest part of analemma)
perihelion_idx = np.argmax(true_anomaly - mean_anomaly)

# --- 4. Setup the Plot ---
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor('black')
ax.set_facecolor('#483d8b') # A dark, dusky Martian sky
ax.set_title("The Analemma on Mars", color='white', pad=15)

# --- 5. Draw Static Scene Elements ---
def setup_scene():
    ax.set_xlim(np.max(equation_of_time_min)+5, np.min(equation_of_time_min)-5) # Reversed
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Equation of Time (minutes from Mean Noon)", color='white')
    ax.set_ylabel("Sun's Declination (°)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Label the special points on the full path
    ax.plot(equation_of_time_min, declination_deg, '-', color='yellow', lw=1.5, alpha=0.3)
    
    ax.plot(equation_of_time_min[summer_solstice_idx], declination_deg[summer_solstice_idx], 'o', c='red')
    ax.text(equation_of_time_min[summer_solstice_idx], declination_deg[summer_solstice_idx]+1, 'N. Summer Solstice\n(Aphelion)', c='w', ha='center', va='bottom')
    
    ax.plot(equation_of_time_min[winter_solstice_idx], declination_deg[winter_solstice_idx], 'o', c='red')
    ax.text(equation_of_time_min[winter_solstice_idx], declination_deg[winter_solstice_idx]-1, 'N. Winter Solstice\n(Perihelion)', c='w', ha='center', va='top')
    
    # Show the line of the "mean sun"
    ax.axvline(0, color='white', linestyle='-.', lw=1, label='Mean Noon')

# --- 6. Initialize Animated Objects ---
sun_plot, = ax.plot([], [], 'o', color='gold', markersize=15, markeredgecolor='orange', zorder=10)
trail_plot, = ax.plot([], [], '-', color='yellow', lw=1.5, zorder=5)
sol_text = ax.text(0.05, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)
eot_text = ax.text(0.05, 0.91, '', color='white', transform=ax.transAxes, fontsize=12)
dec_text = ax.text(0.05, 0.87, '', color='white', transform=ax.transAxes, fontsize=12)

# --- 7. The Animation Update Function ---
def update(frame):
    current_eot = equation_of_time_min[frame]
    current_dec = declination_deg[frame]
    
    sun_plot.set_data([current_eot], [current_dec])
    trail_plot.set_data(equation_of_time_min[:frame+1], declination_deg[:frame+1])
    
    # Update text
    sol_text.set_text(f'Sol: {frame}')
    eot_text.set_text(f"EoT: {current_eot:.1f} min")
    dec_text.set_text(f"Declination: {current_dec:.1f}°")
    
    return sun_plot, trail_plot, sol_text, eot_text, dec_text

# --- 8. Run and Save ---
setup_scene()
ax.legend(loc='upper right', facecolor='black', labelcolor='white', edgecolor='white')
plt.tight_layout(rect=[0, 0, 1, 0.95])
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=30)

# output_filename = 'analemma_mars.mp4'
# print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
# ani.save(output_filename, writer='ffmpeg', fps=30, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
# print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()