import math
import numpy as np
from src.geometry import fit_line, find_min_gradient
import cv2


pixel_to_mm_conversion = 10.2117/2464


def simple_crack_estimator(mask: np.ndarray, num_construction_pts: int = 10, interval: int = 20, skip_top: int = 400, v_length: int = 1000, save_path: str = "plot.png") -> float:

    l_construction_line = {math.inf: 0}
    r_construction_line = {-math.inf: 0}
    min_crack_y = -1
    min_crack_x = None
    min_y, max_y = None, None

    for i, row in enumerate(mask):
        if row.sum() == 0:
            continue
        elif row.sum() > 0:

            active_idcs = np.where(row)[0]
            min_idx = active_idcs.min().item()
            max_idx = active_idcs.max().item()

            if min_y is None:
                min_y = i + skip_top
                max_y = i + v_length

            if i > min_crack_y:
                min_crack_y = i
                min_crack_x = (min_idx + max_idx) / 2

            if i <= max_y and i % interval == 0 and i > min_y:
                left = max(l_construction_line.keys())
                right = min(r_construction_line.keys())

                if min_idx < left:
                    if len(l_construction_line) == num_construction_pts:
                        l_construction_line.pop(left)
                    l_construction_line[min_idx] = i

                if max_idx > right:
                    if len(r_construction_line) == num_construction_pts:
                        r_construction_line.pop(right)
                    r_construction_line[max_idx] = i

    if math.inf in l_construction_line:
        l_construction_line.pop(math.inf)
    if -math.inf in r_construction_line:
        r_construction_line.pop(-math.inf)

    xl, yl = zip(*l_construction_line.items())
    xr, yr = zip(*r_construction_line.items())

    xl, yl = find_min_gradient(np.array(xl), np.array(yl))
    xr, yr = find_min_gradient(np.array(xr), np.array(yr))

    al, bl = fit_line(xl, yl)
    ar, br = fit_line(xr, yr)

    intersect_x = (br - bl) / (al - ar)
    intersect_y = al * intersect_x + bl

    # create a 3-channel BGR image from the binary mask
    gray = (mask > 0).astype(np.uint8) * 255
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    x0, x1 = 0, w - 1

    y0_l = al * x0 + bl
    y1_l = al * x1 + bl
    y0_r = ar * x0 + br
    y1_r = ar * x1 + br

    pt1_l = (int(round(x0)), int(round(y0_l)))
    pt2_l = (int(round(x1)), int(round(y1_l)))
    pt1_r = (int(round(x0)), int(round(y0_r)))
    pt2_r = (int(round(x1)), int(round(y1_r)))

    start_of_crack = (int(round(intersect_x)), int(round(intersect_y)))
    end_of_crack = (int(round(min_crack_x)), int(round(min_crack_y)))

    # cv2.line(img, pt1_l, pt2_l, (0, 255, 0), 10, lineType=cv2.LINE_AA)
    # cv2.line(img, pt1_r, pt2_r, (0, 255, 0), 10, lineType=cv2.LINE_AA)

    # cv2.line(img, (0, min_y), (w, min_y),
    #          (255, 255, 0), 10, lineType=cv2.LINE_AA)
    # cv2.line(img, (0, max_y), (w, max_y),
    #          (255, 255, 0), 10, lineType=cv2.LINE_AA)

    # cv2.circle(img, start_of_crack,
    #            50, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    # cv2.circle(img, end_of_crack,
    #            50, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    # for xi, yi in zip(xl, yl):
    #     cv2.circle(img, (int(xi), int(yi)), radius=20,
    #                color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    # for xi, yi in zip(xr, yr):
    #     cv2.circle(img, (int(xi), int(yi)), radius=20,
    #                color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

    # cv2.imwrite(save_path, img)

    crack_distance = np.sqrt((min_crack_x - intersect_x)
                             ** 2 + (min_crack_y - intersect_y)**2)
    crack_distance_mm = crack_distance.item() * pixel_to_mm_conversion

    return crack_distance_mm, min_y, max_y, pt1_l, pt2_l, pt1_r, pt2_r, start_of_crack, end_of_crack, xl, yl, xr, yr
