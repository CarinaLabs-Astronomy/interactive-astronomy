import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# --- 1. Constants and Parameters ---
NUM_FRAMES = 750
# Orbital parameters
EARTH_ORBIT_RADIUS = 1.0
MARS_ORBIT_RADIUS = 1.52
EARTH_ANGULAR_SPEED = 2 * np.pi / 365.25
MARS_ANGULAR_SPEED = 2 * np.pi / 687.0
MARS_INCLINATION_RAD = np.deg2rad(1.85)

# *** NEW: The crucial parameter to start the animation earlier ***
PHASE_OFFSET_DAYS = 375 # Start ~1 year before opposition to see the full loop

# --- 2. Setup Figure with Two Subplots ---
fig = plt.figure(figsize=(8, 10))
fig.patch.set_facecolor('black')
gs = GridSpec(3, 1, figure=fig)
ax_top = fig.add_subplot(gs[0:2, 0], projection='3d')
ax_top.set_title("Heliocentric View: Tilted Orbits", color='white')
ax_top.set_facecolor('black')
ax_sky = fig.add_subplot(gs[2, 0])
ax_sky.set_title("Geocentric View: The Retrograde Loop", color='white')
ax_sky.set_facecolor('#0b1a38')

# --- 3. Pre-calculate All 3D Paths ---
days = np.arange(NUM_FRAMES)
# *** CORRECTED: Apply the phase offset to start earlier in the cycle ***
time_in_orbit = days - PHASE_OFFSET_DAYS

# Earth's path
earth_angle = EARTH_ANGULAR_SPEED * time_in_orbit
earth_x, earth_y, earth_z = EARTH_ORBIT_RADIUS*np.cos(earth_angle), EARTH_ORBIT_RADIUS*np.sin(earth_angle), np.zeros_like(earth_angle)

# Mars's path
mars_angle = MARS_ANGULAR_SPEED * time_in_orbit
mars_x_flat, mars_y_flat = MARS_ORBIT_RADIUS*np.cos(mars_angle), MARS_ORBIT_RADIUS*np.sin(mars_angle)
Rx = np.array([[1,0,0],[0,np.cos(MARS_INCLINATION_RAD),-np.sin(MARS_INCLINATION_RAD)],[0,np.sin(MARS_INCLINATION_RAD),np.cos(MARS_INCLINATION_RAD)]])
mars_coords_tilted = Rx @ np.vstack([mars_x_flat, mars_y_flat, np.zeros_like(mars_x_flat)])
mars_x, mars_y, mars_z = mars_coords_tilted

# Calculate apparent sky coordinates (this logic remains the same)
dx, dy, dz = mars_x - earth_x, mars_y - earth_y, mars_z - earth_z
apparent_lon_rad = np.arctan2(dy, dx); sky_x_deg = np.rad2deg(np.unwrap(apparent_lon_rad))
dist_xy = np.sqrt(dx**2 + dy**2); apparent_lat_rad = np.arctan2(dz, dist_xy); sky_y_deg = np.rad2deg(apparent_lat_rad)

# --- 4. Setup Static Scene Elements ---
def setup_scenes():
    ax_top.plot([0],[0],[0],'o',color='gold',ms=20,label='Sun')
    # Use the full, un-offset paths for plotting the static orbits
    full_earth_path = EARTH_ORBIT_RADIUS*np.cos(EARTH_ANGULAR_SPEED*days), EARTH_ORBIT_RADIUS*np.sin(EARTH_ANGULAR_SPEED*days), np.zeros_like(days)
    full_mars_path_flat = MARS_ORBIT_RADIUS*np.cos(MARS_ANGULAR_SPEED*days), MARS_ORBIT_RADIUS*np.sin(MARS_ANGULAR_SPEED*days), np.zeros_like(days)
    full_mars_path_tilted = Rx @ np.vstack(full_mars_path_flat)
    
    ax_top.plot(*full_earth_path, ':', color='skyblue', lw=1, label='Earth Orbit (Ecliptic)')
    ax_top.plot(*full_mars_path_tilted, ':', color='orangered', lw=1, label='Mars Orbit (Tilted)')
    
    ax_top.set_box_aspect((1,1,0.5)); ax_top.axis('off'); ax_top.view_init(elev=25, azim=-75)

    np.random.seed(42); star_x=np.random.uniform(np.min(sky_x_deg)-5,np.max(sky_x_deg)+5,100); star_y=np.random.uniform(-3,3,100)
    ax_sky.plot(star_x,star_y,'.',c='w',ms=1,alpha=0.5)
    ax_sky.set_ylim(-2.5,2.5)
    ax_sky.set_xlabel("Apparent Ecliptic Longitude (°)",c='w'); ax_sky.set_ylabel("Latitude (°)",c='w')
    ax_sky.tick_params(colors='white'); ax_sky.invert_xaxis()

# --- 5. Initialize Animated Objects ---
earth_plot, = ax_top.plot([],[],[],'o',c='skyblue',ms=8); mars_plot, = ax_top.plot([],[],[],'o',c='orangered',ms=6)
line_of_sight, = ax_top.plot([],[],[],'-',c='limegreen',lw=1,alpha=0.7)
mars_sky_plot, = ax_sky.plot([],[],'o',c='orangered',ms=10,markeredgecolor='w'); mars_trail, = ax_sky.plot([],[],'-',c='orangered',lw=1.5,alpha=0.8)
day_text = ax_top.text2D(0.05,0.95,'',c='w',transform=ax_top.transAxes,fontsize=12)

# --- 6. The Animation Update Function ---
def update(frame):
    earth_plot.set_data_3d([earth_x[frame]],[earth_y[frame]],[earth_z[frame]])
    mars_plot.set_data_3d([mars_x[frame]],[mars_y[frame]],[mars_z[frame]])
    line_of_sight.set_data_3d([earth_x[frame],mars_x[frame]],[earth_y[frame],mars_y[frame]],[earth_z[frame],mars_z[frame]])
    
    current_sky_x, current_sky_y = sky_x_deg[frame], sky_y_deg[frame]
    mars_sky_plot.set_data([current_sky_x],[current_sky_y])
    mars_trail.set_data(sky_x_deg[:frame+1],sky_y_deg[:frame+1])
    
    day_text.set_text(f'Day: {frame}')
    ax_sky.set_xlim(current_sky_x + 20, current_sky_x - 20)

    return (earth_plot, mars_plot, line_of_sight, 
            mars_sky_plot, mars_trail, day_text)

# --- 7. Run and Save ---
setup_scenes()
ax_top.legend(loc='upper right', facecolor='k', labelcolor='w', edgecolor='w')
plt.tight_layout(pad=2.0)
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=30)

# output_filename = 'retrograde_loop_full_cycle.mp4'
# print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
# ani.save(output_filename, writer='ffmpeg', fps=30, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
# print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()