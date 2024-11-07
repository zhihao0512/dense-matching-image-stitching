import cv2
import flow_loftr
import flow_flowformer
from stitch import image_stitching


if __name__ == '__main__':
    # load loftr and flowformer models
    loftr_model = flow_loftr.load_model()
    flowformer = flow_flowformer.load_model()

    # read input images
    rgb_imgs = []
    rgb_imgs.append(cv2.imread("01.jpg"))
    rgb_imgs.append(cv2.imread("02.jpg"))

    # compute flow map and feature pairs
    [flow_map, flow_viz] = flow_flowformer.flow(rgb_imgs, flowformer)
    [points1, points2, conf_map] = flow_loftr.flow(rgb_imgs, loftr_model)

    # image stitching
    result, seam_result = image_stitching(rgb_imgs, [points1, points2], flow_map, conf_map)
    cv2.imwrite("result.jpg", result)
