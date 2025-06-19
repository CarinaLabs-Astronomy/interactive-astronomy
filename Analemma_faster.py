import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import pytz
from datetime import datetime, timedelta

# --- 1. Constants and Parameters ---
# NEW: Added a step to skip days, making the animation faster.
#      2 = 2x faster, 3 = 3x faster, etc.
STEP_DAYS = 2
YEAR = 2025


# Observer's Location 
OBSERVER_LOCATION = EarthLocation(lat=36.8132*u.deg, lon=14.56663*u.deg, height=200*u.m)  
# Define the fixed time of day for observation (local time)
TIMEZONE = pytz.timezone('Europe/Rome')
OBSERVATION_HOUR = 16
OBSERVATION_MINUTE = 00

# # Observer's Location (e.g., New York City for a nice tilted analemma)
# OBSERVER_LOCATION = EarthLocation(lat=40.7128*u.deg, lon=-74.0060*u.deg, height=10*u.m)  
# # Define the fixed time of day for observation (local time)
# TIMEZONE = pytz.timezone('America/New_York')
# OBSERVATION_HOUR = 12
# OBSERVATION_MINUTE = 0

# --- 2. Pre-calculate All Sun Positions for the Year ---
# MODIFIED: Create a list of datetimes by stepping through the year
start_date = datetime(YEAR, 1, 1, OBSERVATION_HOUR, OBSERVATION_MINUTE, tzinfo=TIMEZONE)
observation_datetimes = [start_date + timedelta(days=i) for i in range(0, 365, STEP_DAYS)]
NUM_FRAMES = len(observation_datetimes) # The number of frames is now based on the stepped dates

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
# To label the points correctly, we still calculate the full path just for finding the markers
full_year_dts = [start_date + timedelta(days=i) for i in range(365)]
full_year_times = Time(full_year_dts)
full_year_altaz = get_sun(full_year_times).transform_to(AltAz(obstime=full_year_times, location=OBSERVER_LOCATION))
full_alts = full_year_altaz.alt.deg
full_azis = full_year_altaz.az.deg

summer_solstice_idx = np.argmax(full_alts)
winter_solstice_idx = np.argmin(full_alts)
vernal_equinox_idx = [dt.date() for dt in full_year_dts].index(datetime(YEAR, 3, 20).date())
autumnal_equinox_idx = [dt.date() for dt in full_year_dts].index(datetime(YEAR, 9, 22).date())

# --- 4. Setup the Plot ---
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#4169E1') #black 
ax.set_facecolor('#4169E1') # Royal blue sky
ax.set_title(f"The Sun's Analemma at {OBSERVATION_HOUR:02d}:{OBSERVATION_MINUTE:02d} Local Time ({STEP_DAYS}x Speed)", color='white')

# --- 5. Draw Static Scene Elements ---
def setup_scene():
    # Set axis limits based on the full path, with some padding
    ax.set_xlim(np.max(full_azis) + 0.5, np.min(full_azis) - 0.5) # Reversed for sky view
    ax.set_ylim(np.min(full_alts) - 10, np.max(full_alts) + 2)
    ax.set_xlabel("Azimuth (°)", color='white'); ax.set_ylabel("Altitude (°)", color='white')
    ax.tick_params(colors='white'); ax.grid(True, linestyle=':', alpha=0.5)

    # Simple horizon line
    ax.axhline(0, color='darkgreen', lw=5)
    ax.fill_between(np.linspace(*ax.get_xlim()), 0, ax.get_ylim()[0], color='darkgreen', zorder=10)
    
    # Label the special points on the path
    ax.plot(full_azis[summer_solstice_idx], full_alts[summer_solstice_idx], 'o', c='red')
    ax.text(full_azis[summer_solstice_idx], full_alts[summer_solstice_idx]+0.5, 'Summer Solstice', c='w', ha='center')
    ax.plot(full_azis[winter_solstice_idx], full_alts[winter_solstice_idx], 'o', c='red')
    ax.text(full_azis[winter_solstice_idx], full_alts[winter_solstice_idx]-1, 'Winter Solstice', c='w', ha='center')
    ax.plot(full_azis[vernal_equinox_idx], full_alts[vernal_equinox_idx], 'o', c='red')
    ax.text(full_azis[vernal_equinox_idx]-0.5, full_alts[vernal_equinox_idx], 'Vernal\nEquinox', c='w', ha='right', va='center')
    ax.plot(full_azis[autumnal_equinox_idx], full_alts[autumnal_equinox_idx], 'o', c='red')
    ax.text(full_azis[autumnal_equinox_idx]+0.5, full_alts[autumnal_equinox_idx], 'Autumnal\nEquinox', c='w', ha='left', va='center')

# --- 6. Initialize Animated Objects ---
sun_plot, = ax.plot([], [], 'o', color='gold', markersize=15, markeredgecolor='orange')
trail_plot, = ax.plot([], [], '-', color='yellow', lw=1.5, alpha=0.7)
date_text = ax.text(0.05, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)

# --- 7. The Animation Update Function ---
def update(frame):
    current_az = azimuths[frame]
    current_alt = altitudes[frame]
    sun_plot.set_data([current_az], [current_alt])
    trail_plot.set_data(azimuths[:frame+1], altitudes[:frame+1])
    current_date = observation_datetimes[frame]
    date_text.set_text(current_date.strftime('%B %d'))
    return sun_plot, trail_plot, date_text

# --- 8. Run and Save ---
setup_scene()
plt.tight_layout()
# The interval can be a normal value now; speed comes from fewer frames.
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=40)

# To display in a window, comment out the ani.save block and uncomment plt.show()
# plt.show()

# To save the animation
output_filename = 'analemma_fast.mp4'
ani.save(output_filename, writer='ffmpeg', fps=30)