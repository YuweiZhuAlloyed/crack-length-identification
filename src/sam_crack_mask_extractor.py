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

    def extract(self):
        agg_results = []

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing images...", total=len(os.listdir(self.input_dir)))

            for image_name in os.listdir(self.input_dir):

                input_path = self.input_dir / image_name
                output_path = self.output_dir / f"mask_{image_name}"
                plot_path = self.output_dir / f"plot_{image_name}.png"

                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.equalizeHist(image)
                image = cv2.medianBlur(image, 39)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                self.predictor.set_image(image)

                input_box = np.array([950, 1000, 1450, 5000])

                masks, confidence, _ = self.predictor.predict(
                    point_coords=np.array([[1200, 1000]]),
                    point_labels=np.array([1]),
                    box=input_box[None, :],
                    multimask_output=False,
                )

                # print(f"Processed {image_name}, saved mask to {output_path}")
                mask = masks[0]
                mask = (mask * 255).astype(np.uint8)

                h, w = mask.shape
                mask_norm = (mask / 255.0).reshape(h, w, 1)
                color = np.array([30/255, 144/255, 255/255])
                img_rgb = image.astype(np.float32) / 255.0

                overlay = img_rgb * (1 - mask_norm) + \
                    mask_norm * color.reshape(1, 1, 3)
                overlay_uint8 = (overlay * 255).astype(np.uint8)

                x0, y0 = int(input_box[0]), int(input_box[1])
                rect_w = int(input_box[2] - input_box[0])
                rect_h = int(input_box[3] - input_box[1])

                overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
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
                    crack_length = simple_crack_estimator(
                        largest_mask,
                        min_y=1100,
                        num_construction_pts=10,
                        save_path=plot_path.as_posix())
                except Exception as e:
                    print(
                        f"Could not estimate crack length for {image_name}: {e}")
                    crack_length = np.nan

                agg_results.append({
                    "image_name": image_name,
                    "seg_confidence": confidence[0],
                    "crack_length_mm": crack_length,

                })

                pl.DataFrame(agg_results).write_csv(
                    self.output_dir / "agg_results.csv")
                progress.update(task, advance=1)
