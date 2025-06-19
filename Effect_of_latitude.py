import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# --- 1. Constants and Parameters ---
NUM_FRAMES = 200
SPHERE_RADIUS = 10
latitudes = np.linspace(0, 90, NUM_FRAMES)

STAR_1_DEC_DEG = 0     # A star on the Celestial Equator
STAR_2_DEC_DEG = 60    # A "circumpolar" star for mid-latitudes

# --- 2. 3D Plot Setup ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("The Effect of Latitude on the Sky", color='white', pad=20)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# --- 3. Helper Function for Rotation ---
def get_tilt_matrix(latitude_deg):
    latitude_rad = np.deg2rad(latitude_deg)
    tilt_angle = latitude_rad - np.pi/2
    return np.array([[1,0,0], [0,np.cos(tilt_angle),-np.sin(tilt_angle)], [0,np.sin(tilt_angle),np.cos(tilt_angle)]])

def rodrigues_rotation(points, axis, angle):
    axis=axis/np.linalg.norm(axis); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return (np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*np.dot(K,K)) @ points

# --- 4. Draw Static Scene (The Horizon) ---
def setup_scene():
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(SPHERE_RADIUS*np.cos(theta), SPHERE_RADIUS*np.sin(theta), 0, c='dimgray', lw=1.5)
    ax.text(0,SPHERE_RADIUS+1.5,0,"N",c='w',ha='center'); ax.text(SPHERE_RADIUS+1.5,0,0,"E",c='w',ha='center')
    ax.text(0,-SPHERE_RADIUS-1.5,0,"S",c='w',ha='center'); ax.text(-SPHERE_RADIUS-1.5,0,0,"W",c='w',ha='center')
    ax.text(0,0,SPHERE_RADIUS+1,"Zenith",c='gray',ha='center',va='center')
    u,v=np.mgrid[0:2*np.pi:30j,0:np.pi:30j]; x_s=SPHERE_RADIUS*np.cos(u)*np.sin(v); y_s=SPHERE_RADIUS*np.sin(u)*np.sin(v); z_s=SPHERE_RADIUS*np.cos(v)
    
    # *** THIS IS THE CORRECTED LINE ***
    ax.plot_wireframe(x_s,y_s,z_s,color='gray',rstride=3,cstride=3,lw=0.5,alpha=0.1)
    
    ax.view_init(elev=20, azim=-75); ax.set_box_aspect((1,1,1)); ax.axis('off')
    
# --- 5. Initialize Animated Objects ---
dynamic_artists = []
latitude_text = fig.text(0.05, 0.95, '', c='w', fontsize=14, transform=fig.transFigure)
ncp_altitude_text = fig.text(0.05, 0.91, '', c='w', fontsize=12, transform=fig.transFigure)
circumpolar_text = fig.text(0.05, 0.87, '', c='w', fontsize=12, transform=fig.transFigure)

# --- 6. The Animation Update Function ---
def update(frame):
    for artist in dynamic_artists: artist.remove()
    dynamic_artists.clear()

    current_lat_deg = latitudes[frame]
    tilt_matrix = get_tilt_matrix(current_lat_deg)
    
    # Draw Tilted Equatorial Grid
    theta = np.linspace(0, 2*np.pi, 100)
    for dec in [-60,-30,0,30,60]:
        dec_rad=np.deg2rad(dec); r_dec=SPHERE_RADIUS*np.cos(dec_rad); z_dec=SPHERE_RADIUS*np.sin(dec_rad)
        pts = tilt_matrix @ np.vstack([r_dec*np.cos(theta), r_dec*np.sin(theta), np.full_like(theta, z_dec)])
        p, = ax.plot(pts[0,:],pts[1,:],pts[2,:], c='r' if dec==0 else 'c', ls='--' if dec==0 else ':', alpha=0.8, lw=1.5 if dec==0 else 1)
        dynamic_artists.append(p)
    for ra in np.arange(0,180,30):
        base_circle=np.vstack([SPHERE_RADIUS*np.cos(theta),np.zeros_like(theta),SPHERE_RADIUS*np.sin(theta)])
        ra_rotated=rodrigues_rotation(base_circle,[0,0,1],np.deg2rad(ra))
        p, = ax.plot(*(tilt_matrix @ ra_rotated), c='c', ls=':', alpha=0.5, lw=1)
        dynamic_artists.append(p)
        
    # Draw Tilted Polar Axis and NCP
    ncp_vec=tilt_matrix@np.array([0,0,1]); axis_pts=np.array([-ncp_vec*SPHERE_RADIUS,ncp_vec*SPHERE_RADIUS]).T
    p, = ax.plot(axis_pts[0], axis_pts[1], axis_pts[2], c='r', lw=2)
    dynamic_artists.append(p)
    p = ax.text(ncp_vec[0]*SPHERE_RADIUS*1.1,ncp_vec[1]*SPHERE_RADIUS*1.1,ncp_vec[2]*SPHERE_RADIUS*1.1,"NCP",c='r',ha='center')
    dynamic_artists.append(p)

    # Draw Tilted Star Paths
    path1_pts=tilt_matrix@np.vstack([SPHERE_RADIUS*np.cos(np.deg2rad(STAR_1_DEC_DEG))*np.cos(theta),SPHERE_RADIUS*np.cos(np.deg2rad(STAR_1_DEC_DEG))*np.sin(theta),np.full_like(theta,SPHERE_RADIUS*np.sin(np.deg2rad(STAR_1_DEC_DEG)))])
    p, = ax.plot(path1_pts[0,:],path1_pts[1,:],path1_pts[2,:],c='yellow',lw=2,label=f'Star at Dec {STAR_1_DEC_DEG}°')
    dynamic_artists.append(p)
    path2_pts=tilt_matrix@np.vstack([SPHERE_RADIUS*np.cos(np.deg2rad(STAR_2_DEC_DEG))*np.cos(theta),SPHERE_RADIUS*np.cos(np.deg2rad(STAR_2_DEC_DEG))*np.sin(theta),np.full_like(theta,SPHERE_RADIUS*np.sin(np.deg2rad(STAR_2_DEC_DEG)))])
    p, = ax.plot(path2_pts[0,:],path2_pts[1,:],path2_pts[2,:],c='lime',lw=2,label=f'Star at Dec {STAR_2_DEC_DEG}°')
    dynamic_artists.append(p)

    # Draw the Circumpolar Region
    dec_boundary_deg = 90 - current_lat_deg
    if dec_boundary_deg < 90:
        dec_range=np.linspace(dec_boundary_deg,90,20); ra_range=np.linspace(0,360,40); ra_grid,dec_grid=np.meshgrid(np.deg2rad(ra_range),np.deg2rad(dec_range))
        x_cap=SPHERE_RADIUS*np.cos(dec_grid)*np.cos(ra_grid); y_cap=SPHERE_RADIUS*np.cos(dec_grid)*np.sin(ra_grid); z_cap=SPHERE_RADIUS*np.sin(dec_grid)
        cap_pts=tilt_matrix @ np.vstack([x_cap.flatten(),y_cap.flatten(),z_cap.flatten()])
        p = ax.plot_surface(*(cap_pts.reshape(3, *x_cap.shape)), color='mediumpurple', alpha=0.4, zorder=0)
        dynamic_artists.append(p)
        
    # Update Text Displays
    latitude_text.set_text(f"Observer's Latitude: {current_lat_deg:.1f}° N")
    ncp_altitude = np.rad2deg(np.arcsin(ncp_vec[2]))
    ncp_altitude_text.set_text(f"NCP Altitude: {ncp_altitude:.1f}°")
    circumpolar_text.set_text(f"Circumpolar if Dec > {dec_boundary_deg:.1f}°")

    if frame == 1:
        circumpolar_patch=Patch(color='mediumpurple',alpha=0.6,label='Circumpolar Region')
        handles,labels=ax.get_legend_handles_labels()
        ax.legend(handles=handles+[circumpolar_patch],loc='upper right',facecolor='k',labelcolor='w',edgecolor='w')

    return dynamic_artists + [latitude_text, ncp_altitude_text, circumpolar_text]

# --- 7. Run and Save ---
setup_scene()
plt.tight_layout(rect=[0, 0, 1, 0.9])
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=60)

# output_filename = 'effect_of_latitude_final.mp4'
# print(f"Saving animation to '{output_filename}'... This will take a few minutes.")
# ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150, progress_callback=lambda i, n: print(f'Encoding frame {i+1} of {n}', end='\r'))
# print(f"\nAnimation saved successfully as '{output_filename}'!")

# To display in a window, comment out the ani.save block and uncomment plt.show()
plt.show()