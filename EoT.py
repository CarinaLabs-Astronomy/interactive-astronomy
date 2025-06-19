import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, ICRS, GeocentricTrueEcliptic
from datetime import datetime

# --- 1. Constants and Parameters ---
NUM_FRAMES = 365
# Create a Time object for every day of the year 2025
start_date = datetime(2025, 1, 1, 12, 0, 0)
times = Time(start_date) + np.arange(NUM_FRAMES) * u.day
days = np.arange(NUM_FRAMES)

# --- 2. Calculate EoT from First Principles using Astropy ---
print("Calculating Equation of Time components from first principles...")
# a) Get the True Sun's position in both Equatorial (RA/Dec) and Ecliptic (Lon/Lat) frames
sun_coords = get_sun(times)
sun_icrs = sun_coords.transform_to(ICRS())      # Equatorial frame
sun_ecl = sun_coords.transform_to(GeocentricTrueEcliptic()) # Ecliptic frame

# b) Calculate the Mean Sun's position
# The Mean Sun moves at a constant rate. Its longitude is the Mean Anomaly.
# First, find time of vernal equinox for the year to sync our Mean Sun.
from astropy.constants import au
from astropy.solar_system import get_body_barycentric_posvel
earth_pos, earth_vel = get_body_barycentric_posvel("earth", times)
sun_pos, sun_vel = get_body_barycentric_posvel("sun", times)
vernal_equinox_time = times[np.argmin(np.abs((earth_pos + sun_pos).z / au))]
# Calculate mean longitude (angle of the Mean Sun)
mean_longitude = 360 * (times - vernal_equinox_time).to(u.day).value / 365.2422 * u.deg

# c) Deconstruct the effects
# Effect of Obliquity: Difference between sun's motion on ecliptic and its projection on equator
# This is (True Ecliptic Longitude) - (True Right Ascension)
effect_obliquity_deg = (sun_ecl.lon - sun_icrs.ra).wrap_at(180*u.deg)

# Effect of Eccentricity: Difference between where a mean sun would be and the true sun
# This is (Mean Ecliptic Longitude) - (True Ecliptic Longitude)
effect_eccentricity_deg = (mean_longitude - sun_ecl.lon).wrap_at(180*u.deg)

# d) The Final Equation of Time: The sum of the two effects
equation_of_time_deg = effect_obliquity_deg + effect_eccentricity_deg

# Convert from degrees to minutes of time (1 degree = 4 minutes)
effect_obliquity_min = effect_obliquity_deg.value * 4
effect_eccentricity_min = effect_eccentricity_deg.value * 4
equation_of_time_min = equation_of_time_deg.value * 4

# --- 3. Setup the Plot with Three Subplots ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.patch.set_facecolor('black'); fig.suptitle("Deconstructing the Equation of Time", color='w', fontsize=16)
for ax, title, color in zip([ax1, ax2, ax3], 
                            ["Effect of Orbital Eccentricity", "Effect of Axial Tilt (Obliquity)", "Combined Equation of Time = (Apparent - Mean Time)"], 
                            ['orangered', 'skyblue', 'lime']):
    ax.set_facecolor('#1a1a1a'); ax.set_ylabel("Minutes", c='w'); ax.set_title(title, c=color)
    ax.tick_params(colors='w'); ax.grid(True, ls=':', alpha=0.3); ax.set_ylim(-18, 18); ax.axhline(0, c='gray', lw=1)
ax3.set_xlabel("Day of the Year", color='white')

# --- 4. Draw Static Scene Elements ---
def setup_scene():
    ax1.plot(days, effect_eccentricity_min, color='orangered', lw=2)
    ax2.plot(days, effect_obliquity_min, color='skyblue', lw=2)
    ax3.plot(days, equation_of_time_min, color='lime', lw=2)
    ax3.text(15,14,'Sundial is Fast',c='lime',ha='left',va='center'); ax3.text(300,-16,'Sundial is Slow',c='lime',ha='center',va='center')

# --- 5. Initialize Animated Objects ---
sweep_lines = [ax.axvline(0, color='yellow', lw=2) for ax in [ax1, ax2, ax3]]
markers = [ax.plot([],[],'o',c='w',ms=8)[0] for ax in [ax1, ax2, ax3]]
texts = [ax.text(0,0,'',c='w',ha='right',va='bottom',bbox=dict(fc='k',alpha=0.7)) for ax in [ax1, ax2, ax3]]

# --- 6. The Animation Update Function ---
def update(frame):
    day = days[frame]
    vals = [effect_eccentricity_min[frame], effect_obliquity_min[frame], equation_of_time_min[frame]]
    
    for i in range(3):
        sweep_lines[i].set_xdata([day, day])
        markers[i].set_data([day], [vals[i]])
        texts[i].set_position((day - 5, vals[i] + 1))
        texts[i].set_text(f'{vals[i]:.1f} min')
    
    texts[2].set_text(f'Sum: {vals[2]:.1f} min') # Override sum text

    return sweep_lines + markers + texts

# --- 7. Run and Save ---
setup_scene()
plt.tight_layout(rect=[0, 0, 1, 0.95])
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=50)

output_filename = 'equation_of_time_rigorous.mp4'
print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
ani.save(output_filename, writer='ffmpeg', fps=25, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()