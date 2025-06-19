import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import pytz
from datetime import datetime, timedelta

# --- 1. Constants and Parameters ---
NUM_FRAMES = 365  # One frame for each day of the year
# Observer's Location (e.g., New York City for a nice tilted analemma)
OBSERVER_LOCATION = EarthLocation(lat=40.7128*u.deg, lon=-74.0060*u.deg, height=10*u.m)
# Define the fixed time of day for observation (local time)
TIMEZONE = pytz.timezone('America/New_York')
OBSERVATION_HOUR = 12
OBSERVATION_MINUTE = 0

# --- 2. Pre-calculate All Sun Positions for the Year ---
# Create a list of datetime objects for every day of the year at the specified local time
start_date = datetime(2025, 1, 1, OBSERVATION_HOUR, OBSERVATION_MINUTE, tzinfo=TIMEZONE)
observation_datetimes = [start_date + timedelta(days=i) for i in range(NUM_FRAMES)]
observation_times = Time(observation_datetimes)

# Get the Sun's position for all those times
sun_coords_gcrs = get_sun(observation_times)

# Define our local AltAz frame
altaz_frame = AltAz(obstime=observation_times, location=OBSERVER_LOCATION)

# Transform the Sun's coordinates to the local AltAz frame
sun_altaz = sun_coords_gcrs.transform_to(altaz_frame)

# Extract the altitude and azimuth arrays for plotting
altitudes = sun_altaz.alt.deg
azimuths = sun_altaz.az.deg

# --- 3. Find Solstice and Equinox Points for Labeling ---
# Find the indices for specific days to label them
summer_solstice_idx = np.argmax(altitudes) # Highest point
winter_solstice_idx = np.argmin(altitudes) # Lowest point
# Find equinoxes by looking for dates closest to March 20 and Sep 22
dates_only = [dt.date() for dt in observation_datetimes]
vernal_equinox_idx = dates_only.index(datetime(2025, 3, 20).date())
autumnal_equinox_idx = dates_only.index(datetime(2025, 9, 22).date())

# --- 4. Setup the Plot ---
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor('black')
ax.set_facecolor('#4169E1') # Royal blue sky
ax.set_title(f"The Sun's Analemma at {OBSERVATION_HOUR:02d}:{OBSERVATION_MINUTE:02d} Local Time", color='white')

# --- 5. Draw Static Scene Elements ---
def setup_scene():
    # Set axis limits based on the calculated path, with some padding
    ax.set_xlim(np.max(azimuths) + 0.5, np.min(azimuths) - 0.5) # Reversed for sky view
    ax.set_ylim(np.min(altitudes) - 2, np.max(altitudes) + 2)
    ax.set_xlabel("Azimuth (°)", color='white'); ax.set_ylabel("Altitude (°)", color='white')
    ax.tick_params(colors='white'); ax.grid(True, linestyle=':', alpha=0.5)

    # Simple horizon line
    ax.axhline(0, color='darkgreen', lw=5)
    ax.fill_between(np.linspace(*ax.get_xlim()), 0, ax.get_ylim()[0], color='darkgreen', zorder=10)
    
    # Label the special points on the path
    ax.plot(azimuths[summer_solstice_idx], altitudes[summer_solstice_idx], 'o', c='red')
    ax.text(azimuths[summer_solstice_idx], altitudes[summer_solstice_idx]+0.5, 'Summer Solstice', c='w', ha='center')
    ax.plot(azimuths[winter_solstice_idx], altitudes[winter_solstice_idx], 'o', c='red')
    ax.text(azimuths[winter_solstice_idx], altitudes[winter_solstice_idx]-1, 'Winter Solstice', c='w', ha='center')
    ax.plot(azimuths[vernal_equinox_idx], altitudes[vernal_equinox_idx], 'o', c='red')
    ax.text(azimuths[vernal_equinox_idx]-0.5, altitudes[vernal_equinox_idx], 'Vernal\nEquinox', c='w', ha='right', va='center')
    ax.plot(azimuths[autumnal_equinox_idx], altitudes[autumnal_equinox_idx], 'o', c='red')
    ax.text(azimuths[autumnal_equinox_idx]+0.5, altitudes[autumnal_equinox_idx], 'Autumnal\nEquinox', c='w', ha='left', va='center')

# --- 6. Initialize Animated Objects ---
sun_plot, = ax.plot([], [], 'o', color='gold', markersize=15, markeredgecolor='orange')
trail_plot, = ax.plot([], [], '-', color='yellow', lw=1.5, alpha=0.7)
date_text = ax.text(0.05, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)

# --- 7. The Animation Update Function ---
def update(frame):
    # Get the position for the current day
    current_az = azimuths[frame]
    current_alt = altitudes[frame]
    
    # Update the Sun's position
    sun_plot.set_data([current_az], [current_alt])
    
    # Update the trail left by the Sun
    trail_plot.set_data(azimuths[:frame+1], altitudes[:frame+1])
    
    # Update the date text
    current_date = observation_datetimes[frame]
    date_text.set_text(current_date.strftime('%B %d'))
    
    return sun_plot, trail_plot, date_text

# --- 8. Run and Save ---
setup_scene()
plt.tight_layout()
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=10)

output_filename = 'analemma.mp4'
print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
ani.save(output_filename, writer='ffmpeg', fps=25, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
# plt.show()