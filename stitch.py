import numpy as np
import cv2 as cv

from feature import calcHomography
from warp import Homography2warp, meshgrid, solveMeshWarping, calcProjectError, mesh_warping
from seam import sigmoidSimilarity, flowSmoothness, duplicationTerm, seamFinding, seamBlending, seamline, drawSeam, \
    point_weight_near_seam, avgBlending
from delaunay import featureTriangle


def calcInOutRect(img1, img2, H):
    pt4 = np.array([[0, 0, 1], [0, img2.shape[0], 1], [img2.shape[1], 0, 1], [img2.shape[1], img2.shape[0], 1]])
    proj_pt4 = np.matmul(np.linalg.inv(H), pt4.transpose())
    proj_pt4 = proj_pt4/proj_pt4[2, :]
    max_xy = np.max(proj_pt4[0:2, :], axis=1)
    max_x = max_xy[0]
    max_y = max_xy[1]
    min_xy = np.min(proj_pt4[0:2, :], axis=1)
    min_x = min_xy[0]
    min_y = min_xy[1]
    offset = np.array([0 - np.min([0, min_x]), 0 - np.min([0, min_y])]).astype(int)
    outrect = np.array([np.min([0, min_x]), np.min([0, min_y]),
                        np.max([img1.shape[1], max_x]), np.max([img1.shape[0], max_y])])
    inrect = np.array([np.max([0, min_x])+offset[0], np.max([0, min_y])+offset[1],
                        np.min([img1.shape[1], max_x])+offset[0], np.min([img1.shape[0], max_y])+offset[1]])
    rect0 = np.array([offset[0], offset[1], img1.shape[1]+offset[0], img1.shape[0]+offset[1]])
    rect1 = np.array([min_x+offset[0], min_y+offset[1], max_x+offset[0], max_y+offset[1]])
    return [outrect.astype(int), inrect.astype(int), rect0.astype(int), rect1.astype(int), offset]


def image_warping_H(rgb_imgs, H, Out_rect, rect0):
    warped_img = Homography2warp(rgb_imgs[1], H, Out_rect)
    mask1 = np.ones(rgb_imgs[1].shape[0:2], dtype=np.float32)
    warped_mask = Homography2warp(mask1, H, Out_rect)
    img0 = np.zeros(warped_img.shape, warped_img.dtype)
    img0[rect0[1]:rect0[3], rect0[0]:rect0[2], :] = rgb_imgs[0]
    mask0 = np.zeros(warped_mask.shape, warped_mask.dtype)
    mask0[rect0[1]:rect0[3], rect0[0]:rect0[2]] = 1
    return [img0, warped_img], [mask0, warped_mask]


def image_warping_mesh(imgs, proj_vertices, x_num, y_num, h, Out_rect, rect0):
    warped_img = mesh_warping(imgs[1], proj_vertices, x_num, y_num, h, Out_rect)
    img0 = np.zeros(warped_img.shape, warped_img.dtype)
    if imgs[0].ndim == 3:
        img0[rect0[1]:rect0[3], rect0[0]:rect0[2], :] = imgs[0]
    else:
        img0[rect0[1]:rect0[3], rect0[0]:rect0[2]] = imgs[0]
    return [img0, warped_img]


def crop_downsize(img, scale):
    img_raw = img[:img.shape[0] // scale * scale, :img.shape[1] // scale * scale]
    img_small = cv.resize(img_raw, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv.INTER_NEAREST)
    return img_small


def crop_upsize(img, scale, image_shape):
    img_large = np.zeros(image_shape, img.dtype)
    img_large[:img.shape[0] * scale, :img.shape[1] * scale] = cv.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv.INTER_NEAREST)
    return img_large


def refine_point_pair_flow(point_pairs, flow):
    for i in range(point_pairs[0].shape[0]):
        p1x = point_pairs[0][i, 0]
        p1y = point_pairs[0][i, 1]
        err = np.linalg.norm(point_pairs[0][i, :]+flow[int(p1y), int(p1x), :]-point_pairs[1][i, :])
        if err < 10:
            point_pairs[1][i, :] = point_pairs[0][i, :]+flow[int(p1y), int(p1x), :]
    return point_pairs


def image_stitching(rgb_imgs, point_pairs, flow, conf_map):
    # reject point pairs with large gap with optical flow
    point_pairs = refine_point_pair_flow(point_pairs, flow)
    # group points by delaunay triangle
    layered_points1, layered_points2, _ = featureTriangle(point_pairs[0], point_pairs[1], 20)
    # calculate global homography and select initial points
    H, _, _ = calcHomography([layered_points1[0], layered_points2[0]], ransac=False)
    _, points1, points2 = calcHomography(point_pairs, ransac=True)

    # calculate boundaries of stitching images
    [Out_rect, In_rect, rect0, rect1, offset] = calcInOutRect(rgb_imgs[0], rgb_imgs[1], H)
    # warp input images by homography
    warped_imgs, warped_masks = image_warping_H(rgb_imgs, H, Out_rect, rect0)

    # upscale confidence map
    conf_map = crop_upsize(conf_map, 8, flow.shape[0:2])
    # calculate smoothness of optical flow field
    similarity_map1 = flowSmoothness(flow, (Out_rect[3]-Out_rect[1], Out_rect[2]-Out_rect[0]), rect0)
    # calculate duplication penalty map
    similarity_map2 = duplicationTerm(flow, similarity_map1, conf_map, H, rect0)
    # calculate the seam finding objective function map
    similarity_map2 = np.maximum(similarity_map1, similarity_map2)+0.005
    # seam estimation
    seam_masks = seamFinding(similarity_map2, warped_masks)
    # seam line mask
    seam = seamline(seam_masks)

    # the size of mesh cell
    h = 40
    # the mesh vertices in the reference image
    [ori_vertices, x_num, y_num] = meshgrid(Out_rect[0], Out_rect[1], Out_rect[2], Out_rect[3], h, h)
    # initialize the projected mesh vertices in the target image
    proj_vertices = ori_vertices
    # initialize point weights by projected error
    points_weight_error = np.ones(points1.shape[0], np.float32)
    # initialize point weights by calculating distances to seam
    point_weight_seam = point_weight_near_seam(points1, seam, rect0, sigma=223.6)
    # initialize the last point weights
    last_points_weight = point_weight_seam.copy()
    # iteratively update mesh warping model and point weights
    for i in range(100):
        # calculate the current point weights
        points_weight = points_weight_error*point_weight_seam
        # until the change of point weights is small
        if ((np.max(np.abs(points_weight - last_points_weight)) < 0.01) & (i > 0)) | (i == 99):
            break
        # solve the projected mesh vertices using point pairs and their weights
        proj_vertices = solveMeshWarping(points1, points2, x_num, y_num, h, Out_rect[0], Out_rect[1], ori_vertices,
                                         H, 5, 0.1, points_weight)
        # calculate point weights by projected error
        points_weight_error = calcProjectError(points1, points2, proj_vertices, x_num, h, Out_rect[0], Out_rect[1], 0.005)
        # update the last point weights
        last_points_weight = points_weight

    # warp input images by the estimated mesh model
    warped_imgs = image_warping_mesh(rgb_imgs, proj_vertices, x_num, y_num, h, Out_rect, rect0)
    mask1 = np.ones(rgb_imgs[1].shape[0:2], dtype=np.float32)
    warped_masks = image_warping_mesh([mask1, mask1.copy()], proj_vertices, x_num, y_num, h, Out_rect, rect0)

    # calculate the photometric error between warped images
    similarity_map0 = sigmoidSimilarity(warped_imgs, warped_masks[0] * warped_masks[1])
    # seam estimation
    seam_masks = seamFinding(similarity_map0+similarity_map2-0.005, warped_masks)
    seam = seamline(seam_masks)
    # image blending by seam
    result = seamBlending(warped_imgs, seam_masks)
    result1 = drawSeam(result, seam)
    return result, result1
