import torch
import cv2
import numpy as np
from LoFTR_module.src.loftr import LoFTR, default_cfg


def load_model():
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("checkpoints/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    return matcher


def flow(imgs, matcher):
    imgs_cuda = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_raw = gray[:gray.shape[0]//8*8, :gray.shape[1]//8*8]
        imgs_cuda.append(torch.from_numpy(gray_raw)[None][None].cuda() / 255.)
    batch = {'image0': imgs_cuda[0], 'image1': imgs_cuda[1]}
    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        conf_map = batch['conf_matrix'].max(dim=2)[0].cpu().numpy().reshape(imgs[0].shape[0]//8, imgs[0].shape[1]//8)
    return [mkpts0, mkpts1, conf_map]
