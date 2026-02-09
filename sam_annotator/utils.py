import base64
import os
import cv2
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from segment_anything import sam_model_registry, SamPredictor
from jupyter_bbox_widget import BBoxWidget

@dataclass_json
@dataclass
class AnnotationMetadata:
    image_name: str
    class_name: str
    bbox: List[int]
    mask_png_path: str
    score: float

class SamAnnotator:
    def __init__(self, checkpoint_path, model_type="vit_h", device="cuda"):
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        self.predictor = SamPredictor(sam)

    def _encode_image(self, filepath):
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def create_widget(self, image_path, output_folder, class_name="barda"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        widget = BBoxWidget()
        widget.image = self._encode_image(image_path)
        
        def on_change(change):
            if not widget.bboxes: return
            
            # 1. Obtener coordenadas
            last_bbox = widget.bboxes[-1]
            x, y, w, h = last_bbox['x'], last_bbox['y'], last_bbox['width'], last_bbox['height']
            input_box = np.array([x, y, x + w, y + h])

            # 2. Inferencia con SAM
            masks, scores, _ = self.predictor.predict(box=input_box, multimask_output=False)
            
            # 3. Guardar Máscara PNG (Formato HRNet: 0 y 255)
            # Nota: Puedes cambiar 255 por 1 si prefieres el formato directo de clase
            mask_name = os.path.basename(image_path).rsplit('.', 1)[0] + "_mask.png"
            mask_path = os.path.join(output_folder, mask_name)
            binary_mask = (masks[0] * 255).astype(np.uint8)
            cv2.imwrite(mask_path, binary_mask)

            # 4. Guardar Metadatos con dataclasses-json
            meta = AnnotationMetadata(
                image_name=os.path.basename(image_path),
                class_name=class_name,
                bbox=[int(x), int(y), int(w), int(h)],
                mask_png_path=mask_path,
                score=float(scores[0])
            )
            
            json_path = mask_path.replace(".png", ".json")
            with open(json_path, 'w') as f:
                f.write(meta.to_json())
                
            print(f"✅ Procesado: {mask_name} | Score: {scores[0]:.2f}")

        widget.observe(on_change, names=['bboxes'])
        return widget