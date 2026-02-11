import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from typing import List

import cv2
import ipywidgets as widgets
import numpy as np
from dataclasses_json import dataclass_json
from google.colab import output
from jupyter_bbox_widget import BBoxWidget
from segment_anything import SamPredictor, sam_model_registry

output.enable_custom_widget_manager()


@dataclass_json
@dataclass
class AnnotationMetadata:
    image_name: str
    class_name: str
    bbox: List[int]
    mask_png_path: str
    score: float


class SamAnnotator:
    def __init__(self, checkpoint_path, model_type, device="cuda"):
        self.device = device
        print(f"Loading SAM model ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        self.predictor = SamPredictor(sam)

        # Image queue state
        self.image_queue = []
        self.output_folder = ""
        self.current_index = None
        self.locked = False

        # Current image state
        self.current_image_path = None
        self.image_rgb = None
        self.final_mask = None
        self.current_metadata_list = []

        # UI Components
        self.bbox_widget = None
        self.btn_check = None
        self.preview_out = None
        self.btn_submit = None
        self.btn_skip = None

    def _encode_image(self, filepath):
        """Encode an image to base64"""
        mime, _ = mimetypes.guess_type(filepath)
        if mime is None:
            mime = "image/png"
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def load_image_queue(self, images_dir, output_folder):
        """Load image file list"""
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_queue = sorted(
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        )

        if not self.image_queue:
            print(f"⚠️ No images found in directory: {images_dir}")

        self.current_index = 0

    def _load_current_data(self):
        """Load current_index image in memory and set it on SAM"""
        if self.current_index >= len(self.image_queue):
            if hasattr(self, "main_layout"):
                self.main_layout.layout.display = "none"
            print("\n" + "=" * 50)
            print("🎉 You have finished annotating all images!")
            print("=" * 50)
            self.current_image_path = None
            return False

        self.current_image_path = self.image_queue[self.current_index]

        image_bgr = cv2.imread(self.current_image_path)
        h, w, _ = image_bgr.shape
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.final_mask = np.zeros((h, w), dtype=np.uint8)
        self.current_metadata_list = []
        self.predictor.set_image(self.image_rgb)
        return True

    def _refresh_widget_content(self):
        """Update widget with current image"""
        if self.current_image_path:
            self.bbox_widget.bboxes = []
            self.bbox_widget.image = self._encode_image(self.current_image_path)
            self.preview_out.layout.display = "none"
        else:
            print("End of dataset.")

    def _next(self):
        self.current_index += 1
        if self._load_current_data():
            self._refresh_widget_content()
        self.locked = False
        self.btn_submit.disabled = False
        self.btn_skip.disabled = False


    def _on_submit_click(self, _):
        if self.locked:
            return

        self.locked = True
        self.btn_submit.disabled = True
        self.btn_skip.disabled = True

        if self.current_image_path and self.current_metadata_list:
            base_name = os.path.basename(self.current_image_path).rsplit(".", 1)[0]
            mask_path = os.path.join(self.output_folder, f"{base_name}_mask.png")
            json_path = os.path.join(self.output_folder, f"{base_name}_mask.json")

            cv2.imwrite(mask_path, self.final_mask)
            for meta in self.current_metadata_list:
                meta.mask_png_path = mask_path

            with open(json_path, "w") as f:
                json_data = [m.to_dict() for m in self.current_metadata_list]
                json.dump(json_data, f, indent=4)

            print(f"✅ Files created for: {base_name}")
        else:
            print("⚠️ No bounding boxes drawn. Nothing was saved!")

        self._next()

    def _on_skip_click(self, _):
        if self.locked:
            return

        self.locked = True
        self.btn_submit.disabled = True
        self.btn_skip.disabled = True

        if self.current_image_path:
            base_name = os.path.basename(self.current_image_path).rsplit(".", 1)[0]
            print(f"⏩ Skipping image: {base_name}")

        self._next()

    def _on_reset_clicked(self, _):
        self.final_mask.fill(0)
        self.bbox_widget.bboxes = []

    def _update_preview_display(self, change=None):
        if not self.btn_check.value or self.current_image_path is None:
            self.preview_out.layout.display = "none"
            return

        overlay = self.image_rgb.copy()
        overlay[self.final_mask > 0] = [0, 255, 0]
        combined = cv2.addWeighted(overlay, 0.5, self.image_rgb, 0.5, 0)

        _, buffer = cv2.imencode(
            ".png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )
        self.preview_out.value = buffer.tobytes()
        self.preview_out.layout.display = "block"

    def _on_bbox_change(self, change):
        current_bboxes = change["new"]
        self.final_mask.fill(0)
        self.current_metadata_list = []

        if not current_bboxes:
            self._update_preview_display()
            return

        for bbox in current_bboxes:
            x, y, bw, bh = (
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"],
            )
            input_box = np.array([x, y, x + bw, y + bh])
            masks, scores, _ = self.predictor.predict(
                box=input_box, multimask_output=False
            )
            current_mask = (masks[0] * 255).astype(np.uint8)
            self.final_mask = cv2.bitwise_or(self.final_mask, current_mask)

            meta = AnnotationMetadata(
                image_name=os.path.basename(self.current_image_path),
                class_name="barda",
                bbox=[int(x), int(y), int(bw), int(bh)],
                mask_png_path="",
                score=float(scores[0]),
            )
            self.current_metadata_list.append(meta)

        self._update_preview_display()

    def start_annotation_session(self, images_dir, output_folder):
        """Start session and load image queue from directory"""
        self.load_image_queue(images_dir, output_folder)

        if not self._load_current_data():
            return "No images to annotate."

        self.bbox_widget = BBoxWidget(hide_buttons=True)
        self.bbox_widget.image = self._encode_image(self.current_image_path)
        self.bbox_widget.classes = []

        self.btn_submit = widgets.Button(
            description="Submit", button_style="success", icon="check"
        )
        self.btn_skip = widgets.Button(
            description="Skip", button_style="warning", icon="forward"
        )
        self.btn_reset = widgets.Button(
            description="Reset Mask", button_style='danger', icon='trash')

        self.btn_submit.on_click(self._on_submit_click)
        self.btn_skip.on_click(self._on_skip_click)
        self.btn_reset.on_click(self._on_reset_clicked)

        self.btn_check = widgets.ToggleButton(
            description="Preview Mask", button_style="info", icon="eye"
        )
        self.preview_out = widgets.Image(format="png")
        self.preview_out.layout.display = "none"

        self.bbox_widget.observe(self._on_bbox_change, names=["bboxes"])
        self.btn_check.observe(self._update_preview_display, names="value")

        controls = widgets.HBox([self.btn_submit, self.btn_skip])
        tools = widgets.HBox([self.btn_reset, self.btn_check])
        preview_box = widgets.VBox(
            [widgets.HTML("<b>Preview:</b>"), self.preview_out],
            layout=widgets.Layout(width="40%", align_items="center")
            )
        bbox_box = widgets.VBox(
            [self.bbox_widget],
            layout=widgets.Layout(width="60%")
            )

        self.main_layout = widgets.VBox(
            [
                controls,
                tools,
                widgets.HBox(
                    [bbox_box, preview_box],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="flex-start",
                        justify_content="space-between"
                    )
                )
            ]
        )
        return self.main_layout
