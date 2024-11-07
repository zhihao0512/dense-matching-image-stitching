import numpy as np
import cv2 as cv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


def transform_H(p, H):
    proj_p = np.dot(H, np.concatenate([p, np.ones([p.shape[0], 1])], axis=1).T)
    proj_p = proj_p[0:2, :]/proj_p[2, :]
    return proj_p.T


def meshgrid(x_min, y_min, x_max, y_max, xh, yh):
    x_vec = np.arange(x_min, x_max+xh, xh)
    y_vec = np.arange(y_min, y_max+yh, yh)
    [x_cor, y_cor] = np.meshgrid(x_vec, y_vec)
    x_cor = x_cor.reshape((-1, 1))
    y_cor = y_cor.reshape((-1, 1))
    xy_cor = np.append(x_cor, y_cor, axis=1)
    return xy_cor, x_vec.shape[0], y_vec.shape[0]


def Homography2warp(img2, H, Out_rect):
    xy_cor, _, _ = meshgrid(Out_rect[0], Out_rect[1], Out_rect[2]-1, Out_rect[3]-1, 1, 1)
    map = transform_H(xy_cor, H).reshape(Out_rect[3]-Out_rect[1], Out_rect[2]-Out_rect[0], 2).astype(np.float32)
    warped_img2 = cv.remap(img2, map[..., 0], map[..., 1], cv.INTER_LINEAR)
    return warped_img2


def alignmentTerm(points1, points2, x_num, h, x_min, y_min, points_weight=None):
    if points_weight is None:
        points_weight = np.ones(points1.shape[0])
    equ_num = 2*points1.shape[0]
    pos = (points1-[x_min, y_min])/[h, h]
    pos0 = np.floor(pos)
    off = pos - pos0
    w4 = off[:, 0]*off[:, 1]
    w3 = off[:, 0]-w4
    w2 = off[:, 1]-w4
    w1 = 1-off[:, 0]-off[:, 1]+w4
    w1 = w1 * points_weight
    w2 = w2 * points_weight
    w3 = w3 * points_weight
    w4 = w4 * points_weight
    vidx1 = pos0[:, 1]*x_num+pos0[:, 0]
    vidx2 = (pos0[:, 1]+1)*x_num+pos0[:, 0]
    vidx3 = pos0[:, 1]*x_num+pos0[:, 0]+1
    vidx4 = (pos0[:, 1]+1)*x_num+pos0[:, 0]+1
    A_col = np.concatenate([vidx1*2, vidx1*2+1, vidx2*2, vidx2*2+1, vidx3*2, vidx3*2+1, vidx4*2, vidx4*2+1])
    A_row = np.concatenate([np.arange(equ_num), np.arange(equ_num), np.arange(equ_num), np.arange(equ_num)])
    A_data = np.concatenate([w1, w1, w2, w2, w3, w3, w4, w4])
    b_row = np.arange(equ_num)
    b_data = np.concatenate([points2[:, 0]*points_weight, points2[:, 1]*points_weight])
    return A_row, A_col, A_data, b_data, equ_num


def smoothTerm(x_num, y_num):
    [xyidx, _, _] = meshgrid(1, 1, x_num-2, y_num-2, 1, 1)
    vidx = xyidx[:, 1]*x_num+xyidx[:, 0]
    vidx1 = (xyidx[:, 1]-1)*x_num+xyidx[:, 0]
    vidx2 = xyidx[:, 1]*x_num+xyidx[:, 0]-1
    vidx3 = (xyidx[:, 1]+1)*x_num+xyidx[:, 0]
    vidx4 = xyidx[:, 1]*x_num+xyidx[:, 0]+1
    equ_num1 = xyidx.shape[0]*2
    A_col = np.concatenate([vidx*2, vidx*2+1, vidx1*2, vidx1*2+1, vidx2*2, vidx2*2+1, vidx3*2, vidx3*2+1, vidx4*2, vidx4*2+1])
    A_row = np.concatenate([np.arange(equ_num1), np.arange(equ_num1), np.arange(equ_num1), np.arange(equ_num1), np.arange(equ_num1)])
    A_data = np.concatenate([np.linspace(1, 1, equ_num1), np.linspace(-0.25, -0.25, 4*equ_num1)])
    [xyidx, _, _] = meshgrid(1, 0, x_num - 2, y_num - 1, 1, y_num - 1)
    vidx = xyidx[:, 1]*x_num+xyidx[:, 0]
    vidx1 = xyidx[:, 1]*x_num+xyidx[:, 0]-1
    vidx2 = xyidx[:, 1]*x_num+xyidx[:, 0]+1
    equ_num2 = xyidx.shape[0]*2
    A_col = np.concatenate([A_col, vidx*2, vidx*2+1, vidx1*2, vidx1*2+1, vidx2*2, vidx2*2+1])
    A_row = np.concatenate([A_row, np.arange(equ_num1, equ_num1 + equ_num2), np.arange(equ_num1, equ_num1 + equ_num2), np.arange(equ_num1, equ_num1 + equ_num2)])
    A_data = np.concatenate([A_data, np.linspace(1, 1, equ_num2), np.linspace(-0.5, -0.5, 2*equ_num2)])
    [xyidx, _, _] = meshgrid(0, 1, x_num - 1, y_num - 2, x_num - 1, 1)
    vidx = xyidx[:, 1]*x_num+xyidx[:, 0]
    vidx1 = (xyidx[:, 1]-1)*x_num+xyidx[:, 0]
    vidx2 = (xyidx[:, 1]+1)*x_num+xyidx[:, 0]
    equ_num3 = xyidx.shape[0]*2
    A_col = np.concatenate([A_col, vidx*2, vidx*2+1, vidx1*2, vidx1*2+1, vidx2*2, vidx2*2+1])
    A_row = np.concatenate([A_row, np.arange(equ_num1 + equ_num2, equ_num1 + equ_num2 + equ_num3),
                            np.arange(equ_num1 + equ_num2, equ_num1 + equ_num2 + equ_num3), np.arange(equ_num1 + equ_num2, equ_num1 + equ_num2 + equ_num3)])
    A_data = np.concatenate([A_data, np.linspace(1, 1, equ_num3), np.linspace(-0.5, -0.5, 2*equ_num3)])
    equ_num = equ_num1+equ_num2+equ_num3
    b_data = np.zeros(equ_num)
    return A_row, A_col, A_data, b_data, equ_num


def globalTerm(v1, H):
    equ_num = v1.shape[0]*2
    proj_v = transform_H(v1, H)
    A_col = np.arange(equ_num)
    A_row = np.arange(equ_num)
    A_data = np.ones(equ_num)
    b_row = np.arange(equ_num)
    b_data = proj_v.reshape(-1)
    return A_row, A_col, A_data, b_data, equ_num


def solveMeshWarping(points1, points2, x_num, y_num, h, x_min, y_min, v1, F, sc, gc, points_weight=None):
    if points_weight is None:
        points_weight = np.ones(points1.shape[0], np.float32)
    [A_row1, A_col1, A_data1, b_data1, equ_num1] = alignmentTerm(points1, points2, x_num, h, x_min, y_min, points_weight)
    [A_row2, A_col2, A_data2, b_data2, equ_num2] = smoothTerm(x_num, y_num)
    [A_row3, A_col3, A_data3, b_data3, equ_num3] = globalTerm(v1, F)
    A_row = np.concatenate([A_row1, A_row2+equ_num1, A_row3+equ_num1+equ_num2])
    A_col = np.concatenate([A_col1, A_col2, A_col3])
    A_data = np.concatenate([A_data1, A_data2*sc, A_data3*gc])
    equ_num = equ_num1+equ_num2+equ_num3
    v_num = x_num*y_num*2
    A_sparse = csr_matrix((A_data.tolist(), (A_row.tolist(), A_col.tolist())), shape=(equ_num, v_num))
    b = np.concatenate([b_data1, b_data2*sc, b_data3*gc]).reshape(-1, 1)
    x = lsqr(A_sparse, b)
    proj_v = x[0].reshape(-1, 2)
    return proj_v


def projectPointbyMesh(p, proj_v, x_num, h, x_min, y_min):
    pos = (p-[x_min, y_min])/[h, h]
    pos0 = np.floor(pos).astype(int)
    off = pos - pos0
    w4 = off[:, 0]*off[:, 1]
    w3 = off[:, 0]-w4
    w2 = off[:, 1]-w4
    w1 = 1-off[:, 0]-off[:, 1]+w4
    vidx1 = pos0[:, 1]*x_num+pos0[:, 0]
    vidx2 = (pos0[:, 1]+1)*x_num+pos0[:, 0]
    vidx3 = pos0[:, 1]*x_num+pos0[:, 0]+1
    vidx4 = (pos0[:, 1]+1)*x_num+pos0[:, 0]+1
    proj_vt = proj_v.T
    proj_p = proj_vt[:, vidx1]*w1+proj_vt[:, vidx2]*w2+proj_vt[:, vidx3]*w3+proj_vt[:, vidx4]*w4
    return proj_p.T


def calcProjectError(points1, points2, proj_v, x_num, h, x_min, y_min, gamma=0.01):
    proj_p = projectPointbyMesh(points1, proj_v, x_num, h, x_min, y_min)
    err = np.sum((proj_p-points2)**2, axis=1)
    points_weight = np.exp(-gamma*err)
    return points_weight


def mesh_warping(img, proj_v, x_num, y_num, h, rect):
    map = proj_v.reshape(y_num, x_num, 2).astype(np.float32)
    map = cv.warpAffine(map, np.array([[h, 0, 0], [0, h, 0]]).astype(np.float32), (rect[2]-rect[0], rect[3]-rect[1]))
    warped_img = cv.remap(img, map[:, :, 0], map[:, :, 1], cv.INTER_LINEAR)
    return warped_img



