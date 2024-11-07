import sys

import numpy as np

sys.path.append('FlowFormer_module/core')

import torch
import torch.nn.functional as F
import cv2
from FlowFormer_module.core.FlowFormer import build_flowformer
from FlowFormer_module.configs.submission import get_cfg
from FlowFormer_module.core.utils.utils import InputPadder
from FlowFormer_module.core.utils import flow_viz

TRAIN_SIZE = [432, 960]


def load_model():
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size}).")
    if image_shape[0] == TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_flow(model, image1, image2, weights=None):
    #print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def flow2match(flow, scale):
    h, w = flow.shape[0:2]
    x_flow = cv2.resize(flow[:flow.shape[0]//scale*scale, :flow.shape[1]//scale*scale, 0], (flow.shape[1]//scale, flow.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
    y_flow = cv2.resize(flow[:flow.shape[0]//scale*scale, :flow.shape[1]//scale*scale, 1], (flow.shape[1]//scale, flow.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
    grid_x, grid_y = np.meshgrid(np.arange(flow.shape[1]//scale), np.arange(flow.shape[0]//scale))
    grid_x = grid_x.astype(np.float32) * scale + scale // 2
    grid_y = grid_y.astype(np.float32) * scale + scale // 2
    proj_x = grid_x + x_flow
    proj_y = grid_y + y_flow
    mask = (proj_x < flow.shape[1]) & (proj_x > 0) & (proj_y < flow.shape[0]) & (proj_y > 0)
    points1 = np.concatenate([grid_x[mask].reshape(-1, 1), grid_y[mask].reshape(-1, 1)], axis=1)
    points2 = np.concatenate([proj_x[mask].reshape(-1, 1), proj_y[mask].reshape(-1, 1)], axis=1)
    return points1, points2


def flow(imgs, model, keep_size=False):
    with torch.no_grad():
        image1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB)
        if not keep_size:
            dsize = compute_adaptive_image_size(imgs[0].shape[0:2])
            image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        weights = None
        flow = compute_flow(model, image1, image2, weights)
        if not keep_size:
            flow = cv2.resize(flow, dsize=(imgs[0].shape[1], imgs[0].shape[0]), interpolation=cv2.INTER_CUBIC)
            flow[..., 0] = flow[..., 0] * imgs[0].shape[1] / dsize[0]
            flow[..., 1] = flow[..., 1] * imgs[0].shape[0] / dsize[1]
        flow_img = flow_viz.flow_to_image(flow)
    return flow, flow_img
