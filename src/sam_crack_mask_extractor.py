# from matplotlib import pyplot as plt
# import matplotlib
from segment_anything import sam_model_registry, SamPredictor
from src.interfaces import ICrackMaskExtractor
import os
from pathlib import Path
import cv2
import numpy as np
import polars as pl
from rich.progress import Progress
from src.crack_length_estimator import simple_crack_estimator
# matplotlib.use("Agg")


class SAMCrackMaskExtractor(ICrackMaskExtractor):

    def __init__(self, input_dir: str, save_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sam_checkpoint = "models/sam_hq_vit_l.pth"
        model_type = "vit_l"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)

        self.prompt_points = np.array([[1200, 1000], [1150, 1000]])

    def extract(self):
        agg_results = []

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing images...", total=len(os.listdir(self.input_dir)))

            for image_name in os.listdir(self.input_dir):

                input_path = self.input_dir / image_name
                output_path = self.output_dir / f"mask_{image_name}"
                plot_path = self.output_dir / f"plot_{image_name}.png"

                original_image = cv2.imread(input_path)

                h, w, _ = original_image.shape

                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                image = cv2.equalizeHist(image)
                image = cv2.medianBlur(image, 39)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                self.predictor.set_image(image)

                input_box = np.array([900, 900, 1500, 5000])

                masks, confidence, _ = self.predictor.predict(
                    point_coords=self.prompt_points,
                    point_labels=np.ones(self.prompt_points.shape[0]),
                    box=input_box[None, :],
                    multimask_output=False,
                )

                mask: np.ndarray = masks[0]
                mask = mask.astype(np.uint8)

                mask_ = np.ones(
                    (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_[:, :, 0] = mask * 255
                mask_[:, :, 1] = mask * 100
                mask_[:, :, 2] = mask * 100

                x0, y0 = int(input_box[0]), int(input_box[1])
                rect_w = int(input_box[2] - input_box[0])
                rect_h = int(input_box[3] - input_box[1])

                overlay_bgr = cv2.addWeighted(original_image, 0.5, mask_,
                                              0.5, 0)

                cv2.circle(overlay_bgr, (1200, 1000), radius=20,
                           color=(255, 0, 0), thickness=-1)
                cv2.rectangle(overlay_bgr, (x0, y0), (x0 + rect_w, y0 + rect_h),
                              color=(0, 255, 0), thickness=2)

                cv2.imwrite(output_path.as_posix(), overlay_bgr)

                contours, _ = cv2.findContours(mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                largest_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(
                    largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                try:
                    crack_distance_mm, min_y, max_y, pt1_l, pt2_l, pt1_r, pt2_r, start_of_crack, end_of_crack, xl, yl, xr, yr = simple_crack_estimator(
                        largest_mask,
                        num_construction_pts=6,
                        interval=100,
                        save_path=plot_path.as_posix())

                    construction_image = original_image.copy()
                    cv2.line(construction_image, pt1_l, pt2_l, (0, 255, 0),
                             7, lineType=cv2.LINE_AA)
                    cv2.line(construction_image, pt1_r, pt2_r, (0, 255, 0),
                             7, lineType=cv2.LINE_AA)

                    cv2.line(construction_image, (0, min_y), (w, min_y),
                             (255, 255, 0), 7, lineType=cv2.LINE_AA)
                    cv2.line(construction_image, (0, max_y), (w, max_y),
                             (255, 255, 0), 7, lineType=cv2.LINE_AA)

                    cv2.circle(construction_image, start_of_crack,
                               30, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(construction_image, end_of_crack,
                               30, (255, 0, 0), -1, lineType=cv2.LINE_AA)

                    for xi, yi in zip(xl, yl):
                        cv2.circle(construction_image, (int(xi), int(yi)), radius=20,
                                   color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                    for xi, yi in zip(xr, yr):
                        cv2.circle(construction_image, (int(xi), int(yi)), radius=20,
                                   color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

                    # cv2.imwrite(plot_path.as_posix(), construction_image)

                    cv2.imwrite(plot_path.as_posix(), np.hstack([
                        original_image, overlay_bgr, construction_image
                    ]), [cv2.IMWRITE_PNG_COMPRESSION, 9])

                except Exception as e:
                    print(
                        f"Could not estimate crack length for {image_name}: {e}")
                    crack_distance_mm = np.nan

                agg_results.append({
                    "image_name": image_name,
                    "seg_confidence": confidence[0],
                    "crack_length_mm": crack_distance_mm,

                })

                pl.DataFrame(agg_results).write_csv(
                    self.output_dir / "agg_results.csv")
                progress.update(task, advance=1)
