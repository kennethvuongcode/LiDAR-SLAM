# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2 as cv
import open3d as o3d
import os
import gtsam

dataset = 20
  
with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load("../data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

# %%
print("Shapes of data or value")
print("encoder counts: " + str(encoder_counts.shape))
print("encoder stamps: " + str(encoder_stamps.shape))
print("lidar_angle_min: " + str(lidar_angle_min))
print("lidar_angle_max: " + str(lidar_angle_max))
print("lidar_angle_increment: " + str(lidar_angle_increment))
print("lidar_range_min: " + str(lidar_range_min))
print("lidar_range_max: " + str(lidar_range_max))
print("lidar_ranges: " + str(lidar_ranges.shape))
print("lidar_stamps: " + str(lidar_stamps.shape))
print("imu_angular_velocity: " + str(imu_angular_velocity.shape))
print("imu_linear_acceleration: " + str(imu_linear_acceleration.shape))
print("imu_stamps: " + str(imu_stamps.shape))
print("disp_stamps: " + str(disp_stamps.shape))
print("rgb_stamps: " + str(rgb_stamps.shape))

# %%
#Part 1 ENCODER AND IMU ODOMETRY
FR, FL, RR, RL = encoder_counts
e_time = encoder_stamps - encoder_stamps[0]
e_int = e_time[1:] - e_time[:-1]
e_int = np.insert(e_int,0,0)

print(e_int.shape)

D_r = ((FR+RR)/2)*0.0022
V_r = D_r/e_int #Assuming there is an initial velocity
V_r[0] = 0

D_l = ((RL+RL)/2)*0.0022
V_l = D_l/e_int
V_l[0] = 0

v = (V_r+V_l)/2

fig,ax = plt.subplots(2,1,figsize=(10,6))

ax[0].plot(e_time,D_r,label = "Right Wheel Distance")
ax[0].plot(e_time,D_l,label = "Left Wheel Distance")
ax[0].set_ylabel("Distance (m)")
ax[0].set_xlabel("Time (s)")
ax[0].set_title("Wheel Displacement Over Time")
ax[0].legend()
ax[0].grid()

ax[1].plot(e_time,V_r,label = "Right Velocity")
ax[1].plot(e_time,V_l,label = "Left Velocity")
ax[1].plot(e_time,v,label = "Velocity")
ax[1].set_ylabel("Velocity (m/s)")
ax[1].set_xlabel("Time (s)")
ax[1].set_title("Velocity Over Time")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# %%
imu_time = imu_stamps - imu_stamps[0]
yaw_ang_velocity = imu_angular_velocity[2]

plt.plot(imu_time, yaw_ang_velocity, label="Yaw Angular Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Yaw Angular Velocity (rad/s)")
plt.title("Yaw Angular Velocity Over Time (IMU)")
plt.grid()


# %%
n = e_time.shape[0]
m = imu_time.shape[0]

x_odo = np.zeros(n)
y_odo = np.zeros(n)
theta = np.zeros(n)

indices = np.searchsorted(imu_time,e_time,side="left")
indices = np.clip(indices,0,len(imu_time)-1)

imu_sync = imu_time[indices]
imu_int = imu_sync[1:] - imu_sync[:-1]

omega_sync = yaw_ang_velocity[indices]

# calculating pose
for i in range(1,n):
    theta[i] = theta[i-1] + omega_sync[i-1]*imu_int[i-1] #assuming pose is 0 at start

    if omega_sync[i-1] * imu_int[i-1] == 0:
        inner = 1  
    else:
        inner = np.sin(omega_sync[i-1] * imu_int[i-1] / 2) / (omega_sync[i-1] * imu_int[i-1] / 2)

    x_odo[i] = x_odo[i-1] + v[i-1] * e_int[i] * inner * np.cos(theta[i-1] + (omega_sync[i-1] * e_int[i] / 2))
    y_odo[i] = y_odo[i-1] + v[i-1] * e_int[i] * inner * np.sin(theta[i-1] + (omega_sync[i-1] * e_int[i] / 2))

T_odo = np.zeros((3,3,n))
for i in range(n):
    T_odo[:,:,i] = np.array([
        [np.cos(theta[i]), -np.sin(theta[i]), x_odo[i]],
        [np.sin(theta[i]),  np.cos(theta[i]), y_odo[i]],
        [0, 0, 1]
    ])


print(T_odo[:,:,0])

plt.figure(figsize=(10, 6))  # Set figure size

plt.plot(e_time, x_odo, label="X Position", color="blue")
plt.plot(e_time, y_odo, label="Y Position", color="red")
plt.plot(imu_sync, theta, label="Theta (Orientation)", color="green")

plt.xlabel("Time (s)")
plt.ylabel("Position / Orientation")
plt.title("Robot Trajectory and Orientation Over Time")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)  # Dotted grid with slight transparency

plt.show()


# %%
#Generating Robot trajectory
plt.plot(x_odo,y_odo)
plt.title("Robot Trajectory from Odometry")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid()


# %%
def sample_points(pc, num_samples=1500):
    """Downsample point cloud by selecting equally spaced points."""
    N = pc.shape[0]
    if N <= num_samples:
        return pc  # If already small, keep it as is
    indices = np.round(np.linspace(0, N - 1, num_samples)).astype(int)
    return pc[indices]

# %%
def disc_yaw(source_pc,target_pc,yaw_steps = 40):
    """Find the best yaw rotation by testing different angles and selecting the lowest ICP error."""
    best_angle = 0
    best_error = float('inf')
    best_transformed_pc = source_pc  

    # Discretize yaw angles from -π to π
    yaw_angles = np.linspace(-np.pi, np.pi, yaw_steps)

    target_avg = np.mean(target_pc,axis=0)
    t_align = target_pc-target_avg

    print(t_align)

    source_avg = np.mean(source_pc,axis=0)
    s_align = source_pc-source_avg

    print(s_align)

    for angle in yaw_angles:
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0,             0,             1]])
        rotated_pc = (R @ s_align.T).T

        tree = scipy.spatial.KDTree(t_align)
        distances, _ = tree.query(rotated_pc)
        error = np.mean(distances**2)

        if error < best_error:
            best_error = error
            best_angle = angle
            best_transformed_pc = rotated_pc

    best_transformed_pc += target_avg 
    print(f"Best yaw angle: {np.degrees(best_angle):.2f}° with error: {best_error:.5f}")

    return best_transformed_pc, best_angle

# %%
def icp(source, target, T_odo=None, max_iter=50):
    '''
    Perform  ICP  to align a source point cloud to a target point cloud
    source: (N,2)
    target: (M,2)
    '''
    error = float('inf')

    if T_odo is None:
        T_odo = np.eye(3)

    source_h = np.hstack((source, np.ones((source.shape[0], 1))))  # (N,3) creating homogeneous points for transformation
    source_h = (T_odo @ source_h.T).T  # Apply initial transformation
    source = source_h[:, :2] 

    pose = T_odo.copy()

    for i in range(max_iter):
        tree = scipy.spatial.KDTree(target)
        dist, idx = tree.query(source)

        # matched_target = target[idx]
        idx = np.clip(idx, 0, target.shape[0] - 1)  # Prevent out-of-bounds indexing
        matched_target = target[idx]


        source_avg = np.mean(source,axis=0)
        target_avg = np.mean(matched_target,axis=0)

        s_align = source-source_avg
        t_align = matched_target-target_avg

        H = s_align.T@t_align

        U,S,Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = target_avg - R @ source_avg

        T = np.eye(3)
        T[:2,:2] = R
        T[:2,2] = t

        source = (R @ source.T).T + t
        error = np.mean(dist)

        pose = T @ pose

    return pose

# %%
T_odo[:,:,0]

# %%
# Scan Matching
# lidar data (1081,N) shape, takes 1081 values per scan, starting at -135 going to 135 degrees step angle 0.25 degrees
# N = 4962
# Given orientation and position from part 1, now match two robot frames knowing this

'''
e_time: odometry data time (4956,)
l_time: lidar data time (4962,)
T_odo: odometry trajectory (3,3,4956)
pc: point cloud in 2d (2,1081,4962)
lt_sync: lidar data time synced to odometry (4956,)
pc_sync: point cloud in 2d synced to odometry (2,1081,4956)
robot_T_lidar: lidar to robot frame (3,3)
T_icp: ICP trajectory (3,3,4956) 
'''
T_odo = T_odo
l_time = lidar_stamps-lidar_stamps[0]

#filtering out values
valid_mask = (lidar_ranges >= lidar_range_min) & (lidar_ranges <= lidar_range_max)
filtered_lidar_ranges = np.where(valid_mask, lidar_ranges, np.nan)  # Use NaN for ignored values

r = filtered_lidar_ranges #(1081,4962) shape
theta = np.radians(np.linspace(-135,135,r.shape[0])).reshape(-1,1)

x_sensor = r * np.cos(theta)
y_sensor = r * np.sin(theta)
pc = np.array([x_sensor,y_sensor]) #(2, 1081, 4962)

# Syncing lidar data with odometry data
indices = np.searchsorted(l_time,e_time,side="left")
indices = np.clip(indices,0,len(l_time)-1)

# Define the origin of robot to be at geometric center 
l_orig_dist = (298.33-(330.2/2))/1000 #in meters

lt_sync = l_time[indices]
pc_sync = pc[:,:,indices]    

robot_T_lidar = np.array([
    [1,0,l_orig_dist],
    [0,1,0],
    [0,0,1]
])

T_icp_lidar = np.zeros((3,3,e_time.shape[0]))
T_icp_lidar[:,:,0] = np.eye(3)

for i in range(1,e_time.shape[0]): #for every scan
    T_delta = np.linalg.inv(T_odo[:,:,i-1]) @ T_odo[:,:,i] # getting relative pose change for initial icp check
    T_transformed = np.linalg.inv(robot_T_lidar)@T_delta@robot_T_lidar # expressing in lidar frame

    target_pc = pc_sync[:,:,i-1].T
    source_pc = pc_sync[:,:,i].T
    
    valid_source = ~np.isnan(source_pc).any(axis=1)
    valid_target = ~np.isnan(target_pc).any(axis=1)
    
    source_pc = source_pc[valid_source]
    target_pc = target_pc[valid_target]

    print("Shape of source_pc:", source_pc.shape)  # Should be (1081, 2)
    print("Shape of target_pc:", target_pc.shape)  # Should be (1081, 2)
    print(i)


    T_delta_icp = icp(source_pc,target_pc,T_transformed)
    
    T_icp_lidar[:,:,i] = T_icp_lidar[:,:,i-1]@T_delta_icp

T_icp = np.zeros((3,3,e_time.shape[0]))
for i in range(e_time.shape[0]):
    T_icp[:,:,i] = T_icp_lidar[:,:,i] @ np.linalg.inv(robot_T_lidar)


# %%
def plot_trajectory(T_icp, T_odo):
    # Extract x, y positions from transformation matrices
    x_icp = T_icp[0, 2, :]  # Extract x-coordinates from ICP
    y_icp = T_icp[1, 2, :]  # Extract y-coordinates from ICP

    x_odo = T_odo[0, 2, :]  # Extract x-coordinates from odometry
    y_odo = T_odo[1, 2, :]  # Extract y-coordinates from odometry

    plt.figure(figsize=(10, 6))
    
    # Plot ICP trajectory
    plt.plot(x_icp, y_icp, label="ICP-Corrected Trajectory", color='blue', linewidth=2)
    
    # Plot Odometry trajectory
    plt.plot(x_odo, y_odo, label="Raw Odometry Trajectory", color='red', linestyle='dashed', linewidth=2)
    
    # Mark Start and End points
    plt.scatter(x_icp[0], y_icp[0], color='green', marker='o', label="Start", s=100)
    plt.scatter(x_icp[-1], y_icp[-1], color='black', marker='x', label="End (ICP)", s=100)
    plt.scatter(x_odo[-1], y_odo[-1], color='orange', marker='x', label="End (Odometry)", s=100)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Robot Trajectory: ICP vs. Odometry")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")  # Keep aspect ratio correct

    plt.show()

# Call function with ICP and Odometry data
plot_trajectory(T_icp, T_odo)



# %%
plt.plot(pc_sync[0,:,0],pc_sync[1,:,0])

# %%
def bresenham(x0, y0, x1, y1):
    """Generate points along a line using Bresenham’s algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

# %%
#Part 3 Occupancy and texture mapping

#creating the occupancy grid
grid_size = 500
resolution = 0.05 #5cm per cell

occupancy_grid_first = np.zeros((grid_size, grid_size), dtype=np.float32) #storing the log odds

#tranforming the lidar sensor points to world coordinates
ff = pc_sync[:,:,0]
ff_homo = np.vstack((ff,np.ones(ff.shape[1])))
ff_trans = T_icp[:,:,0]@ff_homo
ff_trans = np.where(np.isfinite(ff_trans), ff_trans, np.nan)
ff_trans = ff_trans[:, ~np.isnan(ff_trans).any(axis=0)]

robot_origin = np.array([grid_size//2,grid_size//2])
robot_pos = (grid_size//2,grid_size//2)

x_world = ff_trans[0,:]
y_world = ff_trans[1,:]

x_grid = np.round(x_world/resolution + grid_size/2).astype(int)
y_grid = np.round(y_world/resolution + grid_size/2).astype(int)

l_occupied = +2
l_free = -0.5
l_max = 10
l_min = -10

occupancy_grid_first[y_grid, x_grid] = np.clip(occupancy_grid_first[y_grid, x_grid] + l_occupied, l_min, l_max)

for x_end, y_end in zip(x_grid, y_grid):
    free_cells = bresenham(robot_origin[0], robot_origin[1], x_end, y_end)
    
    for x, y in free_cells[:-1]:  # exclude the last point (occupied cell)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            occupancy_grid_first[y, x] = np.clip(occupancy_grid_first[y, x] + l_free, l_min, l_max)

# Display occupancy grid
plt.imshow(occupancy_grid_first, cmap="gray", origin='lower')  # Inverted grayscale
plt.colorbar(label="Normalized Log-Odds (Black = Free Cell, White = Occupied  Cell)")
plt.title("Occupancy Grid Map")
plt.show()


# %%
# Occupancy grid parameters
resolution = 0.05  # 5 cm per grid cell (meters per grid unit)
scale_factor = 1 / resolution  # Convert meters to grid units
grid_size = 800  # Define a large fixed occupancy grid

# Initialize a large occupancy grid (fixed size)
occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

# Define the center of the grid
grid_center_x = grid_size // 3
grid_center_y = grid_size // 3

# Occupancy grid log odds update parameters
l_occupied = +4
l_free = -0.5
l_max = 100
l_min = -100

# Store robot trajectory in world coordinates (meters)
positions_world = []
robot_origin = np.array([0, 0, 1])  # Start in world coordinates (meters)

# Process all time steps
for i in range(e_time.shape[0]):
    print(f"Iteration: {i}")

    # Transform robot position to world coordinates
    curr_origin = np.dot(T_icp[:, :, i], robot_origin)
    positions_world.append(curr_origin[:2])

    # Convert robot position to fixed grid coordinates
    robot_x_grid = np.round(curr_origin[0] * scale_factor + grid_center_x).astype(int)
    robot_y_grid = np.round(curr_origin[1] * scale_factor + grid_center_y).astype(int)

    print(f"Iteration {i}: World Position: {curr_origin[:2]} -> Grid Position: ({robot_x_grid}, {robot_y_grid})")

    # Transform LiDAR points to world coordinates
    ff = pc_sync[:, :, i]
    valid_mask = np.isfinite(ff).all(axis=0)  # Ensure all points are finite
    ff = ff[:, valid_mask]
    ff_homo = np.vstack((ff, np.ones(ff.shape[1])))
    ff_trans = T_icp[:, :, i] @ ff_homo  # Now in world coordinates

    x_world = ff_trans[0, :]
    y_world = ff_trans[1, :]

    # Convert LiDAR points to grid coordinates
    x_grid = np.round(x_world * scale_factor + grid_center_x).astype(int)
    y_grid = np.round(y_world * scale_factor + grid_center_y).astype(int)

    # Ensure indices are within fixed grid boundaries
    valid_mask = (x_grid >= 0) & (x_grid < grid_size) & (y_grid >= 0) & (y_grid < grid_size)
    x_valid = x_grid[valid_mask]
    y_valid = y_grid[valid_mask]

    occupancy_grid[y_valid, x_valid] = np.clip(occupancy_grid[y_valid, x_valid] + l_occupied, l_min, l_max)

    free_x_list = []
    free_y_list = []

    for x_end, y_end in zip(x_valid, y_valid):
        free_cells = np.array(bresenham(robot_x_grid, robot_y_grid, x_end, y_end))[:-1]  # Exclude last occupied cell
        free_x_list.append(free_cells[:, 0])
        free_y_list.append(free_cells[:, 1])

    if free_x_list:
        free_x = np.concatenate(free_x_list)
        free_y = np.concatenate(free_y_list)

        valid_free_mask = (free_x >= 0) & (free_x < grid_size) & (free_y >= 0) & (free_y < grid_size)
        free_x = free_x[valid_free_mask]
        free_y = free_y[valid_free_mask]

        occupancy_grid[free_y, free_x] = np.clip(occupancy_grid[free_y, free_x] + l_free, l_min, l_max)

plt.figure(figsize=(10, 10))
plt.imshow(occupancy_grid, cmap="gray", origin="lower")
plt.colorbar(label="Log Odds")
plt.xlabel("X (grid)")
plt.ylabel("Y (grid)")
plt.title("Fixed Size Occupancy Grid Map")
plt.show()


# %%
# # Occupancy grid parameters
# resolution = 0.05  # 5 cm per grid cell (meters per grid unit)
# scale_factor = 1 / resolution  # Convert meters to grid units
# grid_size = 800  # Define a large fixed occupancy grid

# # Initialize a large occupancy grid (fixed size)
# occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

# # Define the center of the grid
# grid_center_x = grid_size // 3
# grid_center_y = grid_size // 3

# # Occupancy grid log odds update parameters
# l_occupied = +4
# l_free = -0.5
# l_max = 100
# l_min = -100

# # Store robot trajectory in world coordinates (meters)
# positions_world = []
# robot_origin = np.array([0, 0, 1])  # Start in world coordinates (meters)

# # Process all time steps
# for i in range(e_time.shape[0]):
#     print(f"Iteration: {i}")

#     # Transform robot position to world coordinates
#     curr_origin = np.dot(T_odo[:, :, i], robot_origin)
#     positions_world.append(curr_origin[:2])

#     # Convert robot position to fixed grid coordinates
#     robot_x_grid = np.round(curr_origin[0] * scale_factor + grid_center_x).astype(int)
#     robot_y_grid = np.round(curr_origin[1] * scale_factor + grid_center_y).astype(int)

#     print(f"Iteration {i}: World Position: {curr_origin[:2]} -> Grid Position: ({robot_x_grid}, {robot_y_grid})")

#     # Transform LiDAR points to world coordinates
#     ff = pc_sync[:, :, i]
#     valid_mask = np.isfinite(ff).all(axis=0)  # Ensure all points are finite
#     ff = ff[:, valid_mask]
#     ff_homo = np.vstack((ff, np.ones(ff.shape[1])))
#     ff_trans = T_odo[:, :, i] @ ff_homo  # Now in world coordinates

#     x_world = ff_trans[0, :]
#     y_world = ff_trans[1, :]

#     # Convert LiDAR points to grid coordinates
#     x_grid = np.round(x_world * scale_factor + grid_center_x).astype(int)
#     y_grid = np.round(y_world * scale_factor + grid_center_y).astype(int)

#     # Ensure indices are within fixed grid boundaries
#     valid_mask = (x_grid >= 0) & (x_grid < grid_size) & (y_grid >= 0) & (y_grid < grid_size)
#     x_valid = x_grid[valid_mask]
#     y_valid = y_grid[valid_mask]

#     occupancy_grid[y_valid, x_valid] = np.clip(occupancy_grid[y_valid, x_valid] + l_occupied, l_min, l_max)

#     free_x_list = []
#     free_y_list = []

#     for x_end, y_end in zip(x_valid, y_valid):
#         free_cells = np.array(bresenham(robot_x_grid, robot_y_grid, x_end, y_end))[:-1]  # Exclude last occupied cell
#         free_x_list.append(free_cells[:, 0])
#         free_y_list.append(free_cells[:, 1])

#     if free_x_list:
#         free_x = np.concatenate(free_x_list)
#         free_y = np.concatenate(free_y_list)

#         valid_free_mask = (free_x >= 0) & (free_x < grid_size) & (free_y >= 0) & (free_y < grid_size)
#         free_x = free_x[valid_free_mask]
#         free_y = free_y[valid_free_mask]

#         occupancy_grid[free_y, free_x] = np.clip(occupancy_grid[free_y, free_x] + l_free, l_min, l_max)

# plt.figure(figsize=(10, 10))
# plt.imshow(occupancy_grid, cmap="gray", origin="lower")
# plt.colorbar(label="Log Odds")
# plt.xlabel("X (grid)")
# plt.ylabel("Y (grid)")
# plt.title("Fixed Size Occupancy Grid Map")
# plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

def bresenham(x0, y0, x1, y1):
    """Generate points along a line using Bresenham’s algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return np.array(points)

resolution = 0.05  # 5 cm per grid cell (meters per grid unit)
scale_factor = 1 / resolution  # Convert meters to grid units
grid_size = 800  # Define a large fixed occupancy grid

occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
grid_center_x = grid_size // 3
grid_center_y = grid_size // 3

l_occupied = +4
l_free = -0.5
l_max = 100
l_min = -100

positions_world = []
robot_origin = np.array([0, 0, 1])  # Start in world coordinates (meters)

for i in range(e_time.shape[0]):
    print(f"Iteration: {i}")

    curr_origin = np.dot(T_icp[:, :, i], robot_origin)
    positions_world.append(curr_origin[:2])

    robot_x_grid = np.round(curr_origin[0] * scale_factor + grid_center_x).astype(int)
    robot_y_grid = np.round(curr_origin[1] * scale_factor + grid_center_y).astype(int)

    print(f"Iteration {i}: World Position: {curr_origin[:2]} -> Grid Position: ({robot_x_grid}, {robot_y_grid})")

    ff = pc_sync[:, :, i]
    valid_mask = np.isfinite(ff).all(axis=0)  # Ensure all points are finite
    ff = ff[:, valid_mask]

    ff_homo = np.vstack((ff, np.ones(ff.shape[1])))
    ff_trans = T_icp[:, :, i] @ ff_homo  # Now in world coordinates

    ff_trans = np.where(np.isfinite(ff_trans), ff_trans, np.nan)
    ff_trans = ff_trans[:, ~np.isnan(ff_trans).any(axis=0)]

    x_world = ff_trans[0, :]
    y_world = ff_trans[1, :]

    x_grid = np.round(x_world * scale_factor + grid_center_x).astype(int)
    y_grid = np.round(y_world * scale_factor + grid_center_y).astype(int)

    valid_mask = (x_grid >= 0) & (x_grid < grid_size) & (y_grid >= 0) & (y_grid < grid_size)
    x_valid = x_grid[valid_mask]
    y_valid = y_grid[valid_mask]

    occupancy_grid[y_valid, x_valid] = np.clip(occupancy_grid[y_valid, x_valid] + l_occupied, l_min, l_max)

    free_x_list = []
    free_y_list = []

    for x_end, y_end in zip(x_valid, y_valid):
        free_cells = np.array(bresenham(robot_x_grid, robot_y_grid, x_end, y_end))[:-1]  # Exclude last occupied cell
        free_x_list.append(free_cells[:, 0])
        free_y_list.append(free_cells[:, 1])

    if free_x_list:
        free_x = np.concatenate(free_x_list)
        free_y = np.concatenate(free_y_list)

        valid_free_mask = (free_x >= 0) & (free_x < grid_size) & (free_y >= 0) & (free_y < grid_size)
        free_x = free_x[valid_free_mask]
        free_y = free_y[valid_free_mask]

        occupancy_grid[free_y, free_x] = np.clip(occupancy_grid[free_y, free_x] + l_free, l_min, l_max)

positions_world = np.array(positions_world)  # Convert to NumPy array
trajectory_x = np.round(positions_world[:, 0] * scale_factor + grid_center_x).astype(int)
trajectory_y = np.round(positions_world[:, 1] * scale_factor + grid_center_y).astype(int)

plt.figure(figsize=(10, 10))
plt.imshow(occupancy_grid, cmap="gray", origin="lower")
plt.colorbar(label="Log Odds")

plt.plot(trajectory_x, trajectory_y, 'ro-', markersize=2, linewidth=1, label="Robot Trajectory")

plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=6, label="Start")  # Start position (Green)
plt.plot(trajectory_x[-1], trajectory_y[-1], 'bo', markersize=6, label="End")  # End position (Blue)

plt.xlabel("X (grid)")
plt.ylabel("Y (grid)")
plt.title("Fixed Size Occupancy Grid Map with Trajectory")
plt.legend()
plt.show()


# %%
# generate sample pc from disparity images

# IMREAD_UNCHANGED ensures we preserve the precision on depth
disp_img = cv.imread("../data/dataRGBD/Disparity20/disparity20_1.png", cv.IMREAD_UNCHANGED)

# note that cv imports as bgr, so colors may be wrong.
bgr_img = cv.imread("../data/dataRGBD/RGB20/rgb20_1.png")
rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

# from writeup, compute correspondence
height, width = disp_img.shape

dd = np.array(-0.00304 * disp_img + 3.31)
depth = 1.03 / dd

mesh = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')  
i_idxs = mesh[0].flatten()
j_idxs = mesh[1].flatten()

rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051, dtype=np.int32)  # force int for indexing
rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051, dtype=np.int32)

# some may be out of bounds, just clip them
rgb_i = np.clip(rgb_i, 0, height - 1)
rgb_j = np.clip(rgb_j, 0, width - 1)

colors = rgb_img[rgb_i, rgb_j]

# lets visualize the image using our transformation to make sure things look correct (using bgr for opencv)
bgr_colors = bgr_img[rgb_i, rgb_j]
# cv.imshow("color", bgr_colors.reshape((height, width, 3)))

uv1 = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)])
K = np.array([[585.05, 0, 242.94],
              [0, 585.05, 315.84],
              [0, 0, 1]])

# project images to 3d points
points = depth.flatten() * (np.linalg.inv(K) @ uv1)

oRr = np.array([[0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]])

roll, pitch, yaw = 0, 0.48, 0.021

# Rotation Matrices
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
])

R_y = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])

R_z = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
])

# Compute final rotation matrix R = R_z * R_y * R_x
bRk = R_z @ R_y @ R_x
# **Apply Correct Rotation Order (Yaw -> Pitch -> Roll)**
bRk = R_z @ R_y @ R_x  # Z-Y-X order
# we want rRo because we have points in optical frame and want to move them to the regular frame.
points = oRr.T @ points
points = bRk @ points

ground_points = points[2, :] < -.5 # Only keep points below threshold

# Use the boolean mask to filter valid points
filtered_points = points[:, ground_points].T +  [.18,.005,.36] # Ensure shape (N, 3)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Also filter colors to match the valid points
filtered_colors = colors[ground_points]
pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)  # Normalize colors

# Visualize
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# o3d.visualization.draw_geometries([pcd, origin])


# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2 as cv
import open3d as o3d
import os

dataset = 20
  
with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load("../data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

# %%
def find_nearest_time(target_time, source_times):
    """
    Find the closest timestamp in source_times to the given target_time.
    
    :param target_time: The RGB timestamp to match
    :param source_times: Sorted array of timestamps (e.g., pose or disparity times)
    :return: Index of the closest timestamp in source_times
    """
    idx = np.searchsorted(source_times, target_time)
    idx = np.clip(idx, 1, len(source_times) - 1)  # Ensure valid index

    # Compare neighbors to find the closest timestamp
    left_idx = idx - 1
    right_idx = idx
    if abs(source_times[left_idx] - target_time) < abs(source_times[right_idx] - target_time):
        return left_idx
    return right_idx

# %%
def convert_pose_3x3_to_4x4(wTb):
    """
    Convert a 3x3 SE(2) transformation (XY position + yaw rotation) 
    to a 4x4 SE(3) homogeneous transformation for 3D space.
    
    :param wTb: (3x3) 2D transformation matrix (x, y, yaw)
    :return: (4x4) homogeneous transformation matrix
    """
    T = np.eye(4)  # Initialize as identity matrix
    T[:2, :2] = wTb[:2, :2]  # Copy rotation (yaw)
    T[:2, 3] = wTb[:2, 2]    # Copy translation (x, y)
    return T


# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

rgb_time = rgb_stamps-rgb_stamps[0]
disp_time = disp_stamps-disp_stamps[0]

grid_size = 800  
resolution = 0.05  
scale_factor = 1 / resolution  
grid_center_x = grid_size // 3
grid_center_y = grid_size // 3

texture_grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)  
texture_count = np.zeros((grid_size, grid_size), dtype=np.int32)  

K = np.array([[585.05, 0, 242.94],
              [0, 585.05, 315.84],
              [0, 0, 1]])

roll, pitch, yaw = 0, .48, 0.021  

R_x = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])

R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])

R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])

bRk = R_z @ R_y @ R_x  
oRr = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  

kinect_position = np.array([[.18], [.005], [.36]])  

def convert_pose_3x3_to_4x4(wTb):
    T = np.eye(4)  
    T[:2, :2] = wTb[:2, :2]  
    T[:2, 3] = wTb[:2, 2]    
    return T

for rgb_idx, rgb_t in enumerate(rgb_time):
    print(f"Processing RGB frame {rgb_idx} at time {rgb_t}")

    disp_idx = find_nearest_time(rgb_t, disp_time)
    pose_idx = find_nearest_time(rgb_t, e_time)

    disp_path = f"../data/dataRGBD/Disparity20/disparity20_{disp_idx+1}.png"
    disp_img = cv.imread(disp_path, cv.IMREAD_UNCHANGED)
    if disp_img is None:
        print(f"Warning: Disparity image at index {disp_idx} not found!")
        continue

    wTb_3x3 = T_icp[:, :, pose_idx]  
    wTb_4x4 = convert_pose_3x3_to_4x4(wTb_3x3)

    rgb_path = f"../data/dataRGBD/RGB20/rgb20_{rgb_idx+1}.png"
    rgb_img = cv.imread(rgb_path)
    if rgb_img is None:
        print(f"Warning: RGB image at index {rgb_idx} not found!")
        continue
    rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)

    height, width = disp_img.shape
    dd = -0.00304 * disp_img + 3.31
    depth = 1.03 / dd

    j_idxs, i_idxs = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    i_flat, j_flat = i_idxs.flatten(), j_idxs.flatten()
    
    rgb_i = np.clip(((526.37 * i_flat + 19276 - 7877.07 * dd.flatten()) / 585.051).astype(np.int32), 0, height - 1)
    rgb_j = np.clip(((526.37 * j_flat + 16662) / 585.051).astype(np.int32), 0, width - 1)

    colors = rgb_img[rgb_i, rgb_j]  

    uv1 = np.vstack([j_flat, i_flat, np.ones_like(i_flat)])
    points_camera = depth.flatten() * (np.linalg.inv(K) @ uv1)

    points_robot = (bRk @ oRr.T @ points_camera) + kinect_position
    points_homogeneous = np.vstack((points_robot, np.ones((1, points_robot.shape[1]))))
    points_world_homogeneous = wTb_4x4 @ points_homogeneous
    points_world = points_world_homogeneous[:3, :]

    x_grid = np.round(points_world[0, :] * scale_factor + grid_center_x).astype(np.int32)
    y_grid = np.round(points_world[1, :] * scale_factor + grid_center_y).astype(np.int32)

    valid_mask = (x_grid >= 0) & (x_grid < grid_size) & (y_grid >= 0) & (y_grid < grid_size)
    x_valid, y_valid = x_grid[valid_mask], y_grid[valid_mask]
    colors_valid = colors[valid_mask]

    np.add.at(texture_grid, (y_valid, x_valid), colors_valid)
    np.add.at(texture_count, (y_valid, x_valid), 1)

valid_mask = texture_count > 0  
texture_grid[valid_mask] /= texture_count[valid_mask][:, np.newaxis]  
texture_grid_display = np.clip(texture_grid, 0, 255).astype(np.uint8)  

masked_texture_grid = texture_grid_display.copy()
masked_texture_grid[occupancy_grid == 0] = [0, 0, 0]  # Set masked areas to black

plt.figure(figsize=(10, 10))
plt.imshow(masked_texture_grid, origin="lower", alpha=0.9)  
plt.xlabel("X (grid)")
plt.ylabel("Y (grid)")
plt.title("Masked 2D Floor Texture Map")
plt.show()


# %%
#Part 4
import gtsam
from gtsam import NonlinearFactorGraph, Values, Pose2
from gtsam import PriorFactorPose2, BetweenFactorPose2
from gtsam import noiseModel

# %%
# Step 1
# Create an empty factor graph
graph = NonlinearFactorGraph()

# Create a container for initial estimates
initial_estimates = Values()

prior_noise = noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.01])  # [sigma_x, sigma_y, sigma_theta]

# Add the prior factor for pose index = 0
graph.add(PriorFactorPose2(0, Pose2(0.0, 0.0, 0.0), prior_noise))

# Add initial estimate for pose 0
initial_estimates.insert(0, Pose2(0.0, 0.0, 0.0))


# %%
def matrix3x3_to_pose2(T_3x3):
    """
    Convert a 3x3 SE(2) transform to gtsam.Pose2(x, y, theta).
    """
    x = T_3x3[0, 2]
    y = T_3x3[1, 2]
    theta = np.arctan2(T_3x3[1, 0], T_3x3[0, 0])  # from rotation matrix
    return Pose2(x, y, theta)

# %%
n = e_time.shape[0]
odometry_noise = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05])

for i in range(n - 1):
    # T_i, T_ip1 are absolute
    T_i   = T_icp[:, :, i]
    T_ip1 = T_icp[:, :, i+1]

    # Relative transform from i -> i+1
    T_rel = np.linalg.inv(T_i) @ T_ip1
    rel_pose = matrix3x3_to_pose2(T_rel)
    
    # Add BetweenFactor to the graph
    graph.add(BetweenFactorPose2(i, i+1, rel_pose, odometry_noise))
    
    # Add initial guess for pose (i+1) if not already
    if not initial_estimates.exists(i+1):
        # Just use the T_icp absolute as an initial guess
        guess_pose = matrix3x3_to_pose2(T_ip1)
        initial_estimates.insert(i+1, guess_pose)


# %%
print(initial_estimates)x

# %%
def compute_icp_error(source, target, T):
    """
    Compute mean nearest-neighbor distance after applying transform T to 'source'.
    source, target: (N,2) arrays
    T: 3x3 transform
    """
    # Transform source
    source_h = np.hstack((source, np.ones((source.shape[0], 1)))).T  # shape (3, N)
    source_tf = (T @ source_h).T  # shape (N, 3)
    source_tf_2d = source_tf[:, :2]
    
    # KD-tree on target
    from scipy.spatial import KDTree
    tree = KDTree(target)
    dist, _ = tree.query(source_tf_2d)
    error = np.mean(dist)
    return error

# %%
# Step 2a

loop_closure_noise = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05])
K = 10  # fixed interval
thresh = 0.3
for i in range(n):
    print(i)
    j = i - K
    if j < 0:
        continue
    
    # run ICP between scans i and j:
    source_scan = pc_sync[:, :, i].T  # shape (1081, 2)
    valid_source = ~np.isnan(source_scan).any(axis=1)
    source_scan = source_scan[valid_source]

    target_scan = pc_sync[:, :, j].T
    valid_target = ~np.isnan(target_scan).any(axis=1)
    target_scan = target_scan[valid_target]

    
    T_guess = np.linalg.inv(T_icp[:, :, j]) @ T_icp[:, :, i]
    
    T_loop = icp(source_scan, target_scan, T_odo=T_guess, max_iter=50)
    
    # Evaluate ICP error:
    error = compute_icp_error(source_scan, target_scan, T_loop)
    if error < thresh:
        # Convert T_loop_3x3 to a Pose2
        loop_pose = matrix3x3_to_pose2(T_loop)
        
        # Add the loop closure factor
        graph.add(BetweenFactorPose2(j, i, loop_pose, loop_closure_noise))


# %%
num_poses = T_icp.shape[2]

proximity_threshold    = 2.0              # meters
icp_threshold          = 0.3              # ICP error threshold
search_window          = 50               # only check last 50 older poses
max_neighbors          = 2                # each pose adds up to 2 loop closures
orientation_threshold  = np.deg2rad(45)   # skip if > 45 deg difference
max_downsample         = 300              # downsample scans to this many points
# --------------------------------------------------------------------
# Build arrays for pose positions (x, y) and orientations
# --------------------------------------------------------------------
positions    = np.zeros((num_poses, 2))
orientations = np.zeros(num_poses)
for i in range(num_poses):
    Ti = T_icp[:, :, i]
    positions[i, 0] = Ti[0, 2]  # x
    positions[i, 1] = Ti[1, 2]  # y
    # orientation from rotation matrix
    orientations[i] = np.arctan2(Ti[1, 0], Ti[0, 0])

# --------------------------------------------------------------------
# Build the KD-tree with scipy.spatial
# --------------------------------------------------------------------
tree = scipy.spatial.KDTree(positions)

# noise model for loop closures
loop_closure_noise = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05])

# --------------------------------------------------------------------
# Proximity-based loop closure
# --------------------------------------------------------------------
for i in range(num_poses):
    print(i)
    # Find all neighbors within proximity_threshold of pose i
    # query_ball_point returns a list of indices
    neighbors = tree.query_ball_point(positions[i], r=proximity_threshold)

    # remove itself (i) and only consider older poses j < i
    neighbors = [j for j in neighbors if j < i]

    # also limit how far back in time we look
    neighbors = [j for j in neighbors if j >= i - search_window]

    # orientation filter
    valid_neighbors = []
    for j in neighbors:
        dtheta = (orientations[i] - orientations[j] + np.pi) % (2*np.pi) - np.pi
        if abs(dtheta) < orientation_threshold:
            valid_neighbors.append(j)

    # if we have too many, pick only the closest by distance
    if len(valid_neighbors) > max_neighbors:
        dists = np.linalg.norm(positions[valid_neighbors] - positions[i], axis=1)
        sorted_idx = np.argsort(dists)[:max_neighbors]
        valid_neighbors = np.array(valid_neighbors)[sorted_idx]

    # ----------------------------------------------------------------
    # For each neighbor j, run ICP
    # ----------------------------------------------------------------
    for j in valid_neighbors:
        # get scans i and j
        source_scan = pc_sync[:, :, i].T  # shape (M,2)
        valid_src   = ~np.isnan(source_scan).any(axis=1)
        source_scan = source_scan[valid_src]
        if source_scan.shape[0] > max_downsample:
            idxs = np.linspace(0, source_scan.shape[0]-1, max_downsample).astype(int)
            source_scan = source_scan[idxs]

        target_scan = pc_sync[:, :, j].T
        valid_tgt   = ~np.isnan(target_scan).any(axis=1)
        target_scan = target_scan[valid_tgt]
        if target_scan.shape[0] > max_downsample:
            idxs = np.linspace(0, target_scan.shape[0]-1, max_downsample).astype(int)
            target_scan = target_scan[idxs]

        # relative transform from j->i for initial guess
        T_guess = np.linalg.inv(T_icp[:, :, j]) @ T_icp[:, :, i]

        # run ICP
        T_loop = icp(source_scan, target_scan, T_odo=T_guess, max_iter=50)

        # check alignment error
        error = compute_icp_error(source_scan, target_scan, T_loop)
        if error < icp_threshold:
            # add loop closure factor
            loop_pose = matrix3x3_to_pose2(T_loop)
            graph.add(BetweenFactorPose2(j, i, loop_pose, loop_closure_noise))
            # print(f"Loop closure {j}->{i}, error={error:.3f}")


# %%
# # Step 2b

# proximity_threshold = 1.0  # e.g., 2 meters
# icp_threshold = 0.3        # ICP error threshold
# loop_closure_noise = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05])

# num_poses = T_icp.shape[2]

# for i in range(num_poses):
#     print(i)
#     # get (x_i, y_i) from T_icp
#     pose_i = T_icp[:, :, i]
#     x_i = pose_i[0, 2]
#     y_i = pose_i[1, 2]
    
#     # Optionally only look back some window (e.g., last 50 poses) to reduce computation
#     for j in range(max(0, i-100), i):
#         pose_j = T_icp[:, :, j]
#         x_j = pose_j[0, 2]
#         y_j = pose_j[1, 2]

#         dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
#         if dist < proximity_threshold:
#             # Potential loop closure => run ICP

#             # 1. Get scans
#             source_scan = pc_sync[:, :, i].T  # shape (num_points, 2)
#             valid_source = ~np.isnan(source_scan).any(axis=1)
#             source_scan = source_scan[valid_source]

#             target_scan = pc_sync[:, :, j].T
#             valid_target = ~np.isnan(target_scan).any(axis=1)
#             target_scan = target_scan[valid_target]

#             # 2. Initial guess for transform from j->i
#             T_guess = np.linalg.inv(pose_j) @ pose_i

#             # 3. Perform ICP
#             T_loop = icp(source_scan, target_scan, T_odo=T_guess, max_iter=50)

#             # 4. Check ICP error
#             error = compute_icp_error(source_scan, target_scan, T_loop)
#             if error < icp_threshold:
#                 # 5. Add loop closure factor
#                 loop_pose = matrix3x3_to_pose2(T_loop)
#                 graph.add(BetweenFactorPose2(j, i, loop_pose, loop_closure_noise))

# #                 print(f"Loop closure added between {j} <-> {i}: dist={dist:.2f}, icp_error={error:.2f}")

# %%
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(100)

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
result = optimizer.optimize()

optimized_poses = []
for i in range(num_poses):
    pose_i_opt = result.atPose2(i)   # returns gtsam.Pose2 for node i
    optimized_poses.append(pose_i_opt)


# %%
print(optimized_poses)

# %%



