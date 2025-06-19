import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- 1. Constants and Parameters ---
NUM_FRAMES = 360
PRECESSION_PERIOD_YRS = 25772
OBLIQUITY_DEG = 23.439
OBLIQUITY_RAD = np.deg2rad(OBLIQUITY_DEG)

START_YEAR = -2500 # Start in the Age of Aries
END_YEAR = 4000   # End well into the Age of Aquarius
years = np.linspace(START_YEAR, END_YEAR, NUM_FRAMES)

ZODIAC_NAMES = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
                "Libra", "Scorpius", "Sagittarius", "Capricornus", "Aquarius", "Pisces"]
ZODIAC_LON_DEG = np.arange(15, 375, 30)

# --- 2. Setup Figure with a Single 2D Plot ---
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('black')
ax.set_title("The Precession of the Vernal Equinox on an Equatorial Map", color='white', pad=15)
ax.set_facecolor('#0b1a38')

# --- 3. Helper Function ---
def get_equinox_longitude(year):
    """Calculates the Ecliptic Longitude of the Vernal Equinox for a given year."""
    precession_rate_deg_per_yr = 360.0 / PRECESSION_PERIOD_YRS
    # Longitude drifts westward (decreases) from the classical '0 of Aries' point.
    # The crossover from Aries (positive) to Pisces (negative/ >330) happened around 100 BC.
    # We can model this by finding the longitude at a reference point (e.g., J2000) and precessing.
    # A simplified but effective model is to calculate the drift from the Aries/Pisces boundary.
    # This boundary was crossed ~2100 years ago.
    lon_deg = -(precession_rate_deg_per_yr * (year + 100))
    return lon_deg

# --- 4. Setup Static Scene Elements ---
def setup_scene():
    ax.set_xlim(24, 0); ax.set_ylim(-40, 40)
    ax.set_xticks(np.arange(0, 25, 2)); ax.set_yticks(np.arange(-30, 31, 15))
    ax.set_xlabel("Right Ascension (h)", color='w', fontsize=12)
    ax.set_ylabel("Declination (°)", color='w', fontsize=12)
    ax.tick_params(colors='w', labelsize=10); ax.grid(True, linestyle=':', color='cyan', alpha=0.3)
    ax.axhline(0, color='cyan', linestyle='--', lw=2, label='Celestial Equator')

    # Draw static J2000 Zodiac constellation markers
    for name, lon_deg in zip(ZODIAC_NAMES, ZODIAC_LON_DEG):
        coord = SkyCoord(lon=lon_deg*u.deg, lat=0*u.deg, frame='geocentricmeanecliptic', obstime='J2000')
        coord_icrs = coord.transform_to('icrs')
        ra_hour, dec_deg = coord_icrs.ra.hour, coord_icrs.dec.deg
        ax.text(ra_hour, dec_deg + 2, name, color='gold', ha='center', alpha=0.7)
        ax.plot(ra_hour, dec_deg, 'P', color='gold', markersize=8, alpha=0.5)

# --- 5. Initialize Animated Objects ---
ecliptic_plot, = ax.plot([], [], '-', color='#ff6666', lw=2, label='Ecliptic')
equinox_plot, = ax.plot([], [], 'D', color='lime', markersize=12, markeredgecolor='k', label='Vernal Equinox')
year_text = fig.text(0.5, 0.96, '', color='w', fontsize=16, ha='center', transform=fig.transFigure)
age_text = fig.text(0.5, 0.92, '', color='gold', fontsize=14, ha='center', transform=fig.transFigure)

# --- 6. Animation Update Function ---
def update(frame):
    current_year = years[frame]
    
    # The Ecliptic's position on an RA/Dec grid depends on the equinox's RA
    # As the equinox precesses, its RA changes, shifting the whole Ecliptic curve
    equinox_lon_deg = get_equinox_longitude(current_year)
    equinox_ra_deg = equinox_lon_deg
    equinox_ra_hour = (equinox_ra_deg / 15) % 24
    if equinox_ra_hour < 0: equinox_ra_hour += 24
    
    equinox_plot.set_data([equinox_ra_hour], [0])

    ra_hours = np.linspace(0, 24, 400); ra_rad = ra_hours * 15 * np.pi / 180
    equinox_ra_rad = equinox_ra_hour * 15 * np.pi / 180
    dec_rad = np.arcsin(np.sin(OBLIQUITY_RAD) * np.sin(ra_rad - equinox_ra_rad + np.pi))
    dec_deg = np.rad2deg(dec_rad)
    ecliptic_plot.set_data(ra_hours, dec_deg)

    year_str = f"{abs(int(current_year))} {'AD' if current_year >= 0 else 'BC'}"
    year_text.set_text(f"Year: {year_str}")
    
    # *** CORRECTED: Age logic based on ecliptic longitude ***
    # Ensure longitude is in the 0-360 range for indexing
    lon_for_indexing = equinox_lon_deg % 360
    sign_index = int(lon_for_indexing / 30)
    age_text.set_text(f"Age of {ZODIAC_NAMES[sign_index]}")

    return ecliptic_plot, equinox_plot, year_text, age_text

# --- 7. Run and Save ---
setup_scene()
# Place legend outside the plot area
fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.88), 
           facecolor='k', labelcolor='w', edgecolor='w')
plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout for titles

ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=60)

# output_filename = 'precession_equatorial_map_final.mp4'
# print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
# ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
# print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()