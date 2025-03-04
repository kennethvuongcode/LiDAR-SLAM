import numpy as np
from scipy.spatial import KDTree
from utils import read_canonical_model, load_pc, visualize_icp_result

def best_fit_transform(A, B):
    """
    Calculate the least-squares best-fit transformation (rotation + translation)
    between two sets of points A and B.
    
    A, B: (N, 3) numpy arrays representing point clouds.
    
    Returns:
    - R: (3, 3) rotation matrix
    - t: (3, 1) translation vector
    - T: (4, 4) homogeneous transformation matrix
    """

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute cross covariance matrix
    H = A_centered.T @ B_centered

    # Compute Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation matrix (det(R) = 1, no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_B - R @ centroid_A

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def icp(source, target, max_iterations=50, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) registration.

    Parameters:
    - source: (N, 3) numpy array representing source point cloud.
    - target: (M, 3) numpy array representing target point cloud.
    - max_iterations: Maximum number of iterations.
    - tolerance: Convergence threshold.

    Returns:
    - T: (4, 4) Final transformation matrix
    - transformed_source: Transformed source point cloud
    """

    src = np.copy(source)
    prev_error = float('inf')

    for i in range(max_iterations):
        # Build KDTree for nearest neighbor search
        tree = KDTree(target)
        distances, indices = tree.query(src)

        # Get matched points
        matched_target = target[indices]

        # Compute best transformation
        T = best_fit_transform(src, matched_target)

        # Apply transformation
        src = (T[:3, :3] @ source.T).T + T[:3, 3]

        # Check for convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T, src

# Example usage
if __name__ == "__main__":
    obj_name = 'drill'  # 'drill' or 'liq_container'
    num_pc = 4  # Number of point clouds
    source_pc = read_canonical_model(obj_name)

    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)
        pose, transformed_pc = icp(source_pc, target_pc)
        visualize_icp_result(source_pc, target_pc, pose)

