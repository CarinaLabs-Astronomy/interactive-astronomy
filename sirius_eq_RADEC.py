import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

# --- 1. Parameters ---
NUM_FRAMES = 250
SPHERE_RADIUS = 10
TELESCOPE_RADIUS = 0.4
TELESCOPE_LENGTH = 2.0

# --- 2. Astronomical Setup ---
new_delhi_loc = EarthLocation(lat=28.6139*u.deg, lon=77.2090*u.deg, height=216*u.m)
sirius_coord = SkyCoord.from_name('Sirius')
start_time_utc = Time('2025-01-02 12:30:00')
observation_times = start_time_utc + np.linspace(0, 12, NUM_FRAMES) * u.hour

# --- 3. Calculate All Necessary Coordinates ---
altaz_frame = AltAz(obstime=observation_times, location=new_delhi_loc)
sirius_altaz = sirius_coord.transform_to(altaz_frame)
lst = observation_times.sidereal_time('mean', new_delhi_loc.lon)
sirius_ha = (lst - sirius_coord.ra).wrap_at(360*u.deg)
sirius_dec = sirius_coord.dec

# Filter for visible part of the path, keeping Angle objects for display
visible_mask = sirius_altaz.alt > 0*u.deg
altitudes_deg = sirius_altaz.alt[visible_mask].deg
azimuths_deg = sirius_altaz.az[visible_mask].deg
hour_angles_rad = sirius_ha[visible_mask].radian
declination_rad = sirius_dec.radian
visible_ha = sirius_ha[visible_mask]
visible_lst = lst[visible_mask]

# --- 4. 3D Plot Setup ---
fig = plt.figure(figsize=(12, 11))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Equatorial Mount Tracking in a Horizon System", color='white', pad=20)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# --- 5. Helper Functions & Global Transformation ---
def altaz_to_cartesian(alt, az, r=1.0):
    alt_rad, az_rad = np.deg2rad(alt), np.deg2rad(az)
    x = r * np.cos(alt_rad) * np.sin(az_rad); y = r * np.cos(alt_rad) * np.cos(az_rad); z = r * np.sin(alt_rad)
    return np.array([x, y, z])

def rodrigues_rotation(points, axis, angle):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)) @ points

latitude_rad = new_delhi_loc.lat.radian
tilt_angle = latitude_rad - np.pi/2
GLOBAL_TILT_MATRIX = np.array([[1, 0, 0], [0, np.cos(tilt_angle), -np.sin(tilt_angle)], [0, np.sin(tilt_angle), np.cos(tilt_angle)]])
POLAR_AXIS_VEC = GLOBAL_TILT_MATRIX @ np.array([0, 0, 1])

# --- 6. Pre-calculate Path ---
star_path_cartesian = altaz_to_cartesian(altitudes_deg, azimuths_deg, SPHERE_RADIUS)

# --- 7. Draw Static Scene ---
def setup_scene():
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(SPHERE_RADIUS*np.cos(theta), SPHERE_RADIUS*np.sin(theta), 0, c='dimgray', lw=1.5)
    ax.text(0, SPHERE_RADIUS+1.5, 0, "N", c='w', ha='center'); ax.text(SPHERE_RADIUS+1.5, 0, 0, "E", c='w', ha='center')
    ax.text(0, -SPHERE_RADIUS-1.5, 0, "S", c='w', ha='center'); ax.text(-SPHERE_RADIUS-1.5, 0, 0, "W", c='w', ha='center')
    
    phi = np.linspace(0, np.pi, 100); y_m = SPHERE_RADIUS * np.cos(phi); z_m = SPHERE_RADIUS * np.sin(phi)
    ax.plot(np.zeros_like(y_m), y_m, z_m, color='magenta', lw=1.5, ls='-.')
    
    # Draw Equatorial Grid and Paths... (condensed for clarity)
    for dec in [-60, -30, 30, 60]:
        pts = GLOBAL_TILT_MATRIX @ np.vstack([SPHERE_RADIUS*np.cos(np.deg2rad(dec))*np.cos(theta),SPHERE_RADIUS*np.cos(np.deg2rad(dec))*np.sin(theta),np.full_like(theta,SPHERE_RADIUS*np.sin(np.deg2rad(dec)))]); ax.plot(pts[0,:],pts[1,:],pts[2,:],c='c',alpha=0.3,lw=0.75)
    for ra in np.arange(0,180,30):
        pts=GLOBAL_TILT_MATRIX @ rodrigues_rotation(np.vstack([SPHERE_RADIUS*np.cos(theta),np.zeros_like(theta),SPHERE_RADIUS*np.sin(theta)]),[0,0,1],np.deg2rad(ra)); ax.plot(pts[0,:],pts[1,:],pts[2,:],c='c',alpha=0.3,lw=0.75)
    eq_pts=GLOBAL_TILT_MATRIX @ np.vstack([SPHERE_RADIUS*np.cos(theta),SPHERE_RADIUS*np.sin(theta),np.zeros_like(theta)]); ax.plot(eq_pts[0,:],eq_pts[1,:],eq_pts[2,:],'--',c='r',lw=1.5)
    sirius_dec_pts=GLOBAL_TILT_MATRIX @ np.vstack([SPHERE_RADIUS*np.cos(declination_rad)*np.cos(theta),SPHERE_RADIUS*np.cos(declination_rad)*np.sin(theta),np.full_like(theta,SPHERE_RADIUS*np.sin(declination_rad))]); ax.plot(sirius_dec_pts[0,:],sirius_dec_pts[1,:],sirius_dec_pts[2,:],c='g',lw=2)
    axis_line = np.array([-POLAR_AXIS_VEC*(SPHERE_RADIUS+2),POLAR_AXIS_VEC*(SPHERE_RADIUS+2)]).T; ax.plot(axis_line[0],axis_line[1],axis_line[2],c='r',lw=1.5)
    ncp_pos=POLAR_AXIS_VEC*(SPHERE_RADIUS+0.5); ax.text(ncp_pos[0],ncp_pos[1],ncp_pos[2],"NCP",c='r',ha='center')

    # *** NEW: Static Data Display (Top-Left) ***
    ax.text2D(0.02, 0.95, "Target: Sirius (Fixed Coords)", transform=ax.transAxes, color='white', weight='bold')
    ra_str = sirius_coord.ra.to_string(unit=u.hour, sep=('h ', 'm ', 's'), pad=True, precision=0)
    dec_str = sirius_coord.dec.to_string(unit=u.deg, sep=('° ', "' ", '"'), precision=0)
    ax.text2D(0.02, 0.90, f"RA: {ra_str}", transform=ax.transAxes, color='white')
    ax.text2D(0.02, 0.85, f"Dec: {dec_str}", transform=ax.transAxes, color='white')

    ax.view_init(elev=25, azim=-65); ax.set_box_aspect((1, 1, 1)); ax.axis('off')

# --- 8. Initialize Animated Objects ---
star_plot, = ax.plot([], [], [], '*', c='yellow', markersize=15, zorder=10)
path_plot, = ax.plot([], [], [], color='yellow', lw=2.5, zorder=6)
ray_plot, = ax.plot([], [], [], '-', c='yellow', lw=1, alpha=0.6, zorder=7)
telescope_surface, dec_axis_line = None, None
# Dynamic Text Objects
ax.text2D(0.02, 0.25, "Observer's Sky Clock (Live)", transform=ax.transAxes, color='white', weight='bold')
lst_text = ax.text2D(0.02, 0.20, '', transform=ax.transAxes, c='w', fontsize=12)
ha_text = ax.text2D(0.02, 0.15, '', transform=ax.transAxes, c='w', fontsize=12)
ra_calc_text = ax.text2D(0.02, 0.10, '', transform=ax.transAxes, c='w', fontsize=12)

# --- 9. Animation Update Function ---
def update(frame):
    global telescope_surface, dec_axis_line
    
    star_x, star_y, star_z = star_path_cartesian[:, frame]
    star_plot.set_data_3d([star_x], [star_y], [star_z])
    path_plot.set_data_3d(star_path_cartesian[0,:frame+1], star_path_cartesian[1,:frame+1], star_path_cartesian[2,:frame+1])
    
    # Update Dynamic Text Display
    lst_str = visible_lst[frame].to_string(unit=u.hour, sep=('h ', 'm ', 's'), pad=True, precision=0)
    ha_str = visible_ha[frame].to_string(unit=u.hour, sep=('h ', 'm ', 's'), pad=True, precision=0)
    calculated_ra = (visible_lst[frame] - visible_ha[frame]).wrap_at(360*u.deg)
    ra_calc_str = calculated_ra.to_string(unit=u.hour, sep=('h ', 'm ', 's'), pad=True, precision=0)
    lst_text.set_text(f"LST: {lst_str}")
    ha_text.set_text(f"Hour Angle: {ha_str}")
    ra_calc_text.set_text(f"RA (LST-HA): {ra_calc_str}")

    # Update Telescope and Ray... (condensed for clarity)
    if telescope_surface: telescope_surface.remove()
    if dec_axis_line: dec_axis_line.remove()
    current_ha_rad = hour_angles_rad[frame]
    N_POINTS_CIRCLE=20; cyl_theta=np.linspace(0,2*np.pi,N_POINTS_CIRCLE); z_cyl=np.linspace(0,TELESCOPE_LENGTH,2); cyl_theta_grid,z_grid=np.meshgrid(cyl_theta,z_cyl)
    tube_template = np.vstack([(TELESCOPE_RADIUS*np.cos(cyl_theta_grid)).flatten(),(TELESCOPE_RADIUS*np.sin(cyl_theta_grid)).flatten(),z_grid.flatten()])
    dec_axis_template = np.array([[-1.5,1.5],[0,0],[0,0]])
    tube_dec_locked = rodrigues_rotation(tube_template,[1,0,0],np.pi/2 - declination_rad)
    assembly_ha_tracked = rodrigues_rotation(np.hstack([tube_dec_locked,dec_axis_template]),[0,0,1],-current_ha_rad)
    final_assembly_pts = GLOBAL_TILT_MATRIX @ assembly_ha_tracked
    final_tube_pts = final_assembly_pts[:,:tube_dec_locked.shape[1]]; final_dec_axis_pts = final_assembly_pts[:,tube_dec_locked.shape[1]:]
    tube_X,tube_Y,tube_Z = final_tube_pts[0].reshape(2,N_POINTS_CIRCLE),final_tube_pts[1].reshape(2,N_POINTS_CIRCLE),final_tube_pts[2].reshape(2,N_POINTS_CIRCLE)
    telescope_surface = ax.plot_surface(tube_X,tube_Y,tube_Z,color='silver',zorder=5)
    dec_axis_line, = ax.plot(final_dec_axis_pts[0],final_dec_axis_pts[1],final_dec_axis_pts[2],color='lightgray',lw=8,zorder=4)
    tip_template=np.array([[0],[0],[TELESCOPE_LENGTH]]); tip_dec_locked=rodrigues_rotation(tip_template,[1,0,0],np.pi/2 - declination_rad); tip_ha_tracked=rodrigues_rotation(tip_dec_locked,[0,0,1],-current_ha_rad); tip_final=GLOBAL_TILT_MATRIX @ tip_ha_tracked
    ray_plot.set_data_3d([tip_final[0,0],star_x],[tip_final[1,0],star_y],[tip_final[2,0],star_z])

    return star_plot, ha_text, ray_plot, path_plot, lst_text, ra_calc_text

# --- 10. Run and Save ---
setup_scene()
num_visible_frames = len(altitudes_deg)
ani = FuncAnimation(fig, update, frames=num_visible_frames, blit=False, interval=50)

output_filename = 'sirius_final_definitive_view.mp4'
print(f"Saving animation to '{output_filename}'... This may take a few minutes.")
ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
# plt.show()