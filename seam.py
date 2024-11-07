import numpy as np
import cv2 as cv

import ctypes as C
gridcut = C.cdll.LoadLibrary("./maxflow_v303.dll")
drawline = C.cdll.LoadLibrary("./line_python.dll")
from warp import meshgrid, transform_H


def sigmoidSimilarity(imgs, mask):
    img1 = imgs[0].astype(np.float32)/255
    img2 = imgs[1].astype(np.float32)/255
    img_dif = np.sqrt(((img1[..., 0]-img2[..., 0])**2+(img1[..., 1]-img2[..., 1])**2+(img1[..., 2]-img2[..., 2])**2))
    beta = 30
    para_alpha = 0.2
    img_dif_sig = 1/(1+np.exp(beta*(-img_dif+para_alpha)))
    img_dif_sig = img_dif_sig*mask
    return img_dif_sig


def flowSmoothness(flow, image_shape, rect0):
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    dx = cv.Laplacian(flow_x, cv.CV_32F)
    dy = cv.Laplacian(flow_y, cv.CV_32F)
    score = (1.001-np.exp(-0.001*(dx**2+dy**2)))
    flow_smooth = np.zeros(image_shape, score.dtype)
    flow_smooth[rect0[1]:rect0[1]+flow.shape[0], rect0[0]:rect0[0]+flow.shape[1]] = score
    return flow_smooth


def duplicationTerm(flow, smooth_map, conf, H, rect0):
    # find pixel matches by optical flow
    ori_pixels, _, _ = meshgrid(0, 0, flow.shape[1]-1, flow.shape[0]-1, 1, 1)
    flow_pixels = flow.reshape(-1, 2)+ori_pixels
    in_pixel_idx = (flow_pixels[:, 0] >= 0) & (flow_pixels[:, 0] < flow.shape[1]) & (flow_pixels[:, 1] >= 0) \
                   & (flow_pixels[:, 1] < flow.shape[0])
    p1 = ori_pixels[in_pixel_idx]
    p2 = flow_pixels[in_pixel_idx]
    # reproject points in image2 into image1
    proj_p2 = transform_H(p2, np.linalg.inv(H))
    # calculate weights for pixels by smoothness and matching confidence
    score = np.zeros_like(smooth_map)
    weight = (smooth_map[rect0[1]:rect0[1]+flow.shape[0], rect0[0]:rect0[0]+flow.shape[1]]*conf).reshape(-1)[in_pixel_idx]
    score0 = (score * 255).astype(np.uint8)
    for i in range(p1.shape[0]):
        # ignore pixel matches with low weights
        if weight[i] > 0.01:
            # draw lines between all pixel pairs, choose the maximum pixel value
            # python
            # line_map = np.zeros_like(score)
            # cv.line(line_map, (p1[i, 0]+rect0[0], p1[i, 1]+rect0[1]), (int(proj_p2[i, 0]+rect0[0]), int(proj_p2[i, 1])+rect0[1]), float(weight[i]))
            # score = np.maximum(score, line_map)
            # c++
            drawline.aggerate_line(score0.ctypes.data_as(C.POINTER(C.c_uint8)),
                                   int(score.shape[0]), int(score.shape[1]),
                                   np.array([p1[i, 0]+rect0[0], p1[i, 1]+rect0[1]]).ctypes.data_as(C.POINTER(C.c_int)),
                                   np.array([int(proj_p2[i, 0]+rect0[0]), int(proj_p2[i, 1])+rect0[1]]).ctypes.data_as(C.POINTER(C.c_int)),
                                   np.array([weight[i]*255]).astype(np.uint8).ctypes.data_as(C.POINTER(C.c_uint8)), 8)
    score = score0.astype(np.float32) / 255
    return score


def seamFinding(loss, masks):
    # overlapping region of two masks
    mask0 = masks[0].astype(np.uint8)
    mask1 = masks[1].astype(np.uint8)
    overlap = mask0 & mask1
    # calculate the minimum enclosing rectangle of overlapping region
    min_x = np.min(np.where(np.sum(overlap, axis=0) > 0))
    max_x = np.max(np.where(np.sum(overlap, axis=0) > 0))
    min_y = np.min(np.where(np.sum(overlap, axis=1) > 0))
    max_y = np.max(np.where(np.sum(overlap, axis=1) > 0))
    in_rect = loss[min_y:max_y+1, min_x:max_x+1].copy()
    # crop the label of minimum enclosing rectangle
    label = (mask0-overlap)*1+(mask1-overlap)*2
    label += 2 * ((mask0 - cv.erode(mask0, None)) & cv.erode(mask1, None))
    label += 1 * ((mask1 - cv.erode(mask1, None)) & cv.erode(mask0, None))
    label = label.astype(np.float32)
    in_label = label[min_y:max_y+1, min_x:max_x+1].copy()
    # seam estimation by maxflow cutting algorithm
    gridcut.calculateSeam(int(max_y+1-min_y), int(max_x+1-min_x), in_rect.ctypes.data_as(C.POINTER(C.c_float)), in_label.ctypes.data_as(C.POINTER(C.c_float)))
    label[min_y:max_y + 1, min_x:max_x + 1] = in_label.copy()
    seam_masks = []
    seam_masks.append((label == 1).astype(np.uint8)*255)
    seam_masks.append((label == 2).astype(np.uint8)*255)
    return seam_masks


def seamBlending(imgs, masks):
    n = len(imgs)
    result = np.zeros(imgs[0].shape, np.uint8)
    for i in range(n):
        m = masks[i]/255
        result = result+imgs[i]*cv.cvtColor(m.astype(np.uint8), cv.COLOR_GRAY2BGR)
    return result


def avgBlending(imgs, seam=None):
    n = len(imgs)
    result = np.zeros(imgs[0].shape, np.uint8)
    for i in range(n):
        result = result+imgs[i]//2
    if seam is not None:
        result[..., 0] = seam
    return result


def seamline(seam_masks):
    seam = cv.dilate(seam_masks[0], None)
    seam = seam & seam_masks[1]
    return seam


def drawSeam(result, seam):
    result1 = np.copy(result)
    result1 = result1.reshape(-1, 3)
    seam1 = seam.reshape(-1)
    result1[seam1.astype(bool), 0] = 0
    result1[seam1.astype(bool), 1] = 0
    result1[seam1.astype(bool), 2] = 255
    result2 = result1.reshape(result.shape)
    return result2


def seamSensorMap(seam, sigma):
    dt = cv.distanceTransform(255-seam, cv.DIST_L2, 3)
    map = np.exp(-0.5*dt*dt/(sigma*sigma))
    map[map < 0.01] = 0
    return map


def point_weight_near_seam(points1, seam, rect0, sigma=20):
    sample_region = seamSensorMap(seam, sigma)
    sample_region = sample_region[rect0[1]:rect0[3], rect0[0]:rect0[2]]
    point_weight = []
    for i in range(points1.shape[0]):
        point_weight.append(sample_region[int(points1[i, 1]), int(points1[i, 0])])
    point_weight = np.array(point_weight)
    return point_weight



