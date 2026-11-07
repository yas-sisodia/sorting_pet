import os
import time
from typing import Dict, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Conveyor configuration
CONVEYOR_SPEED_MPS = 1.5      
CAMERA_TO_ACTUATOR_M = 0.60   

class SortingInferenceEngine:
    def __init__(self, model_name: str = "yolov8n"):
        pt_path = f"{model_name}.pt"
        self.onnx_path = f"{model_name}.onnx"

        # Auto-export to ONNX on Mac if missing
        if not os.path.exists(self.onnx_path):
            print(f"[*] Exporting {pt_path} to ONNX format...")
            model = YOLO(pt_path)
            model.export(format="onnx", imgsz=640, simplify=True)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        # Optimize for Apple Silicon (M1/M2/M3)
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(self.onnx_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = self.session.get_inputs()[0].shape[2]  

        self.classes: Dict[int, str] = {
            39: "PET_Bottle",   
            0: "HDPE_Flake",
            1: "PP_Container",
            2: "LDPE_Film"
        }

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_top = (self.img_size - nh) // 2
        pad_left = (self.img_size - nw) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0), scale, (pad_left, pad_top)

    def calculate_actuator_delay(self, inference_ms: float) -> float:
        flight_time_ms = (CAMERA_TO_ACTUATOR_M / CONVEYOR_SPEED_MPS) * 1000.0
        return max(0.0, flight_time_ms - inference_ms)

    def infer(self, frame: np.ndarray) -> Tuple[dict, float]:
        blob, scale, (pad_left, pad_top) = self.preprocess(frame)

        t_start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        preds = np.squeeze(outputs[0]).T
        scores = np.max(preds[:, 4:], axis=1)
        class_ids = np.argmax(preds[:, 4:], axis=1)

        # Filter ONLY for Bottle (39) or Cup (41) in the COCO dataset
        valid_mask = np.isin(class_ids, [39, 41])
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(scores[valid_mask])]
            
            conf = float(scores[best_idx])
            class_id = int(class_ids[best_idx])
            
            # YOLO outputs center_x, center_y, width, height (on 640x640)
            cx, cy, w, h = preds[best_idx, :4]
            
            # Scale coordinates back to your Mac's native camera resolution
            x1 = int((cx - w / 2 - pad_left) / scale)
            y1 = int((cy - h / 2 - pad_top) / scale)
            x2 = int((cx + w / 2 - pad_left) / scale)
            y2 = int((cy + h / 2 - pad_top) / scale)
            
            label = "PET_Bottle" if class_id == 39 else "Plastic_Cup"
        else:
            # Fallback if no bottle is detected
            conf, x1, y1, x2, y2 = 0.0, 0, 0, 0, 0
            label = "No_Target"

        delay_ms = self.calculate_actuator_delay(latency_ms)
        action = "AIR_BLAST_VALVE_1" if conf > 0.40 else "PASS_CONVEYOR"

        result = {
            "material_class": label,
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "inference_latency_ms": round(latency_ms, 2),
            "actuator_trigger_delay_ms": round(delay_ms, 2),
            "action": action
        }
        return result, latency_ms