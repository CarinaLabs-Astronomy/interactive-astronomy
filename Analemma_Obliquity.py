import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import pytz
from datetime import datetime, timedelta

# --- 1. Constants and Parameters ---
NUM_FRAMES = 365
OBSERVER_LOCATION = EarthLocation(lat=40.7128*u.deg, lon=-74.0060*u.deg, height=10*u.m)
TIMEZONE = pytz.timezone('America/New_York')
OBSERVATION_HOUR = 12
OBSERVATION_MINUTE = 0

# --- 2. Pre-calculate All Sun Positions for the Year ---
start_date = datetime(2025, 1, 1, OBSERVATION_HOUR, OBSERVATION_MINUTE, tzinfo=TIMEZONE)
observation_datetimes = [start_date + timedelta(days=i) for i in range(NUM_FRAMES)]
observation_times = Time(observation_datetimes)
sun_coords_gcrs = get_sun(observation_times)
altaz_frame = AltAz(obstime=observation_times, location=OBSERVER_LOCATION)
sun_altaz = sun_coords_gcrs.transform_to(altaz_frame)
altitudes = sun_altaz.alt.deg
azimuths = sun_altaz.az.deg

# --- 3. Find Key Points and Calculate Obliquity ---
summer_solstice_idx = np.argmax(altitudes)
winter_solstice_idx = np.argmin(altitudes)
# The "Mean Sun" altitude is the average altitude over the year
mean_sun_altitude = np.mean(altitudes)
# The obliquity is the max deviation from this mean
measured_obliquity = altitudes[summer_solstice_idx] - mean_sun_altitude

# --- 4. Setup the Plot ---
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor('black')
ax.set_facecolor('#4169E1')
ax.set_title(f"The Sun's Analemma at {OBSERVATION_HOUR:02d}:{OBSERVATION_MINUTE:02d} Local Time", color='white')

# --- 5. Draw Static Scene Elements ---
def setup_scene():
    ax.set_xlim(np.max(azimuths) + 1, np.min(azimuths) - 1); ax.set_ylim(np.min(altitudes) - 2, np.max(altitudes) + 4)
    ax.set_xlabel("Azimuth (°)", color='white'); ax.set_ylabel("Altitude (°)", color='white')
    ax.tick_params(colors='white'); ax.grid(True, linestyle=':', alpha=0.5)
    ax.axhline(0, color='darkgreen', lw=5); ax.fill_between(np.linspace(*ax.get_xlim()), 0, ax.get_ylim()[0], color='darkgreen', zorder=10)
    
    # *** NEW: Draw the Mean Sun's path and Obliquity measurements ***
    # 1. Mean Sun Altitude Line
    ax.axhline(mean_sun_altitude, color='white', linestyle='-.', lw=1.5, label='Mean Sun Altitude (No Tilt)')
    
    # 2. Lines showing deviation at solstices
    ss_az, ss_alt = azimuths[summer_solstice_idx], altitudes[summer_solstice_idx]
    ws_az, ws_alt = azimuths[winter_solstice_idx], altitudes[winter_solstice_idx]
    ax.plot([ss_az, ss_az], [mean_sun_altitude, ss_alt], ':', color='red', lw=2)
    ax.plot([ws_az, ws_az], [mean_sun_altitude, ws_alt], ':', color='red', lw=2)

    # 3. Text labels for the obliquity
    ax.text(ss_az + 0.2, mean_sun_altitude + measured_obliquity / 2, f"+{measured_obliquity:.1f}°\n(Obliquity)", 
            color='white', ha='left', va='center', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(ws_az + 0.2, mean_sun_altitude - measured_obliquity / 2, f"-{measured_obliquity:.1f}°\n(Obliquity)", 
            color='white', ha='left', va='center', bbox=dict(facecolor='black', alpha=0.5))

    ax.legend(loc='upper right', facecolor='black', labelcolor='white', edgecolor='white')

# --- 6. Initialize Animated Objects ---
sun_plot, = ax.plot([], [], 'o', color='gold', markersize=15, markeredgecolor='orange')
trail_plot, = ax.plot([], [], '-', color='yellow', lw=1.5, alpha=0.7)
# Text displays
date_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes, fontsize=12)
alt_text = ax.text(0.02, 0.91, '', color='white', transform=ax.transAxes, fontsize=12)
dev_text = ax.text(0.02, 0.87, '', color='white', transform=ax.transAxes, fontsize=12)

# --- 7. The Animation Update Function ---
def update(frame):
    current_az = azimuths[frame]
    current_alt = altitudes[frame]
    
    sun_plot.set_data([current_az], [current_alt])
    trail_plot.set_data(azimuths[:frame+1], altitudes[:frame+1])
    
    # Update text
    current_date = observation_datetimes[frame]
    date_text.set_text(current_date.strftime('%B %d'))
    alt_text.set_text(f"Sun Altitude: {current_alt:.2f}°")
    
    # *** NEW: Show deviation from the mean dynamically ***
    deviation = current_alt - mean_sun_altitude
    sign = '+' if deviation >= 0 else '-'
    dev_text.set_text(f"Deviation: {sign}{abs(deviation):.2f}°")
    
    return sun_plot, trail_plot, date_text, alt_text, dev_text

# --- 8. Run and Save ---
setup_scene()
plt.tight_layout(rect=[0, 0, 1, 0.95])
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=50)

# output_filename = 'analemma_obliquity.mp4'
# print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
# ani.save(output_filename, writer='ffmpeg', fps=25, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
# print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()