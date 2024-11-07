import cv2 as cv
import numpy as np
from homo_ransac import ransac_homography


def calcHomography(all_point_pairs, ransac=True, t=0.5):
    points1 = all_point_pairs[0]
    points2 = all_point_pairs[1]
    if ransac:
        # [H, mask] = cv.findHomography(points1, points2, cv.RANSAC, 100.0, 2000)
        H, inliers = ransac_homography(points1, points2, threshold=t)
        return [H, points1[inliers], points2[inliers]]
    else:
        H = cv.findHomography(points1, points2)[0]
        return [H, points1, points2]


