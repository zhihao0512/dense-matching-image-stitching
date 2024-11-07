import numpy as np
from scipy.spatial import Delaunay


def triangle2Edge(tri):
    edge_idx = np.concatenate([tri, tri], axis=1).reshape(-1, 2)
    edge_idx2 = np.sort(edge_idx, axis=1)
    edge = np.unique(edge_idx2, axis=0)
    return edge


def groupbyEdge(edge, num):
    group_idx = np.zeros(num, dtype=np.int)-1
    vlist = []
    group_num = 0
    for i in range(edge.shape[0]):
        if group_idx[edge[i, 0]] == -1 and group_idx[edge[i, 1]] == -1:
            group_idx[edge[i, 0]] = group_num
            group_idx[edge[i, 1]] = group_num
            vlist.append([])
            vlist[group_num].append(edge[i, 0])
            vlist[group_num].append(edge[i, 1])
            group_num += 1
        elif group_idx[edge[i, 0]] != -1 and group_idx[edge[i, 1]] == -1:
            idx = group_idx[edge[i, 0]]
            group_idx[edge[i, 1]] = idx
            vlist[idx].append(edge[i, 1])
        elif group_idx[edge[i, 0]] == -1 and group_idx[edge[i, 1]] != -1:
            idx = group_idx[edge[i, 1]]
            group_idx[edge[i, 0]] = idx
            vlist[idx].append(edge[i, 0])
        elif group_idx[edge[i, 0]] != group_idx[edge[i, 1]]:
            idxs = group_idx[edge[i, :]]
            dst_idx = np.min(idxs)
            src_idx = np.max(idxs)
            for v in vlist[src_idx]:
                group_idx[v] = dst_idx
                vlist[dst_idx].append(v)
            vlist[src_idx] = []
    vlist1 = []
    for vs in vlist:
        if len(vs) > 0:
            vlist1.append(vs)
    for i in range(group_idx.shape[0]):
        if group_idx[i] == -1:
            vlist1.append([i])
    return vlist1


def featureTriangle(points1, points2, th):
    # find delaunay edges among points
    points = (points1+points2)/2
    devision = Delaunay(points)
    tri = devision.simplices.copy()
    coplan = devision.coplanar.copy()
    edge = triangle2Edge(tri)
    edge = np.append(edge, coplan[:, [0, 2]], axis=0)
    # select edges by distance
    edge_points1 = points1[edge.reshape(-1).tolist(), :].reshape(-1, 4)
    edge_points2 = points2[edge.reshape(-1).tolist(), :].reshape(-1, 4)
    edge_point_dis = edge_points1-edge_points2
    edge_dif = np.linalg.norm(edge_point_dis[:, 0:2]-edge_point_dis[:, 2:4], axis=1)
    sel_edge = edge[edge_dif < th, :]
    # divide point set into subsets, points connected with edges are grouped into the same subset
    grouped_vertex = groupbyEdge(sel_edge, points.shape[0])
    # divide points into groups, sorted by point number
    layered_points1 = []
    layered_points2 = []
    grouped_num = []
    for vl in grouped_vertex:
        grouped_num.append(len(vl))
    sort_idx = np.argsort(np.array(grouped_num))[::-1]
    for i in range(sort_idx.shape[0]):
        layered_points1.append(points1[grouped_vertex[sort_idx[i]]])
        layered_points2.append(points2[grouped_vertex[sort_idx[i]]])
    return layered_points1, layered_points2, sel_edge



