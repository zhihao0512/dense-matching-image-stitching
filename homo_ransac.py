import numpy as np


def normalize_points(pts):
    """
    Normalize points so that the mean is 0 and the standard deviation is sqrt(2)
    :param pts: Array of shape (n, 2), representing the coordinates of the points
    :return: Transformation matrix of shape (3, 3) for normalizing point coordinates
    """
    mean = np.mean(pts, axis=0)
    pts_norm = pts - mean
    std = np.mean(np.linalg.norm(pts_norm, axis=1))
    T = np.array([[np.sqrt(2)/std, 0, -np.sqrt(2)/std*mean[0]],
                  [0, np.sqrt(2)/std, -np.sqrt(2)/std*mean[1]],
                  [0, 0, 1]])
    return T


def compute_homography(pts1, pts2, w=None):
    """
    Compute the homography matrix
    :param pts1: Array of shape (n, 2), representing points in the first image
    :param pts2: Array of shape (n, 2), representing corresponding points in the second image
    :return: 3x3 homography matrix
    """
    n = pts1.shape[0]
    if w is None:
        w = np.ones(n)
    A = np.zeros((2*n, 9))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2*i] = np.array([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])*w[i]
        A[2*i+1] = np.array([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])*w[i]
    _, _, V = np.linalg.svd(A)
    h = V[-1,:] / V[-1,-1]
    H = h.reshape((3,3))
    return H


def ransac_homography(pts1, pts2, n_iters=2000, threshold=0.5):
    """
    Estimate the homography matrix using the RANSAC algorithm
    :param pts1: Array of shape (n, 2), representing points in the first image
    :param pts2: Array of shape (n, 2), representing corresponding points in the second image
    :param n_iters: Number of iterations
    :param threshold: Threshold for determining whether a point is an inlier
    :return: 3x3 homography matrix and indices of the inliers
    """
    T1 = normalize_points(pts1)
    T2 = normalize_points(pts2)
    pts1_norm = np.dot(T1, np.vstack((pts1.T, np.ones(pts1.shape[0])))).T[:, :2]
    pts2_norm = np.dot(T2, np.vstack((pts2.T, np.ones(pts2.shape[0])))).T[:, :2]

    best_H = None
    best_inliers = None
    max_inliers = 0
    for i in range(n_iters):
        # Randomly select four points
        indices = np.random.choice(pts1_norm.shape[0], 4, replace=False)
        H = compute_homography(pts1_norm[indices], pts2_norm[indices])
        # Compute the reprojection error for each point
        errors = np.sqrt(np.sum((pts2_norm - np.dot(H, np.vstack((pts1_norm.T, np.ones(pts1_norm.shape[0]))))[:2, :].T)**2, axis=1))
        # Find inliers
        inliers = np.where(errors < threshold)[0]
        # Update optimal solution
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_H = compute_homography(pts1_norm[inliers], pts2_norm[inliers])
            best_inliers = inliers
    best_H = np.dot(np.linalg.inv(T2), np.dot(best_H, T1))
    return best_H, best_inliers
