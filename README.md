# ♻️ Real-Time Optical Plastic Sorter (Edge AI Prototype)

A fast, lightweight Computer Vision prototype built to detect and sort plastic waste (like PET bottles and HDPE) on high-speed industrial conveyor belts. 

It tracks objects in real time and calculates the exact millisecond delay needed for an air-jet nozzle to blast recyclable plastics off the belt into the correct bin.

---

## 💡 The Problem & Solution

* **The Problem:** In automated recycling plants, conveyor belts move at speeds around **1.5 m/s**. If an AI model takes too long to detect a plastic item, the bottle has already moved past the air valve, causing mis-sorting or sorting jams.
* **The Solution:** This project uses an edge-optimized **YOLO model running on ONNX Runtime** decoupled across two parallel threads:
  1. **Thread 1 (Camera Ingestion):** Grabs frames continuously without freezing or buffering.
  2. **Thread 2 (Inference & Actuator Timing):** Runs inference in **~11 ms**, leaving over **380 ms** of safety buffer for the physical air blast to fire accurately.

---

## ⚡ Performance Benchmark (Apple Silicon / Mac)

* **Inference Latency:** `~11.5 ms` (~86 FPS)
* **Conveyor Speed Calibration:** `1.5 m/s`
* **Sensor-to-Valve Distance:** `0.60 m` (60 cm)
* **Pneumatic Trigger Delay Buffer:** `~388 ms`

### Sample Output (Ready for Industrial PLCs)
```json
{
  "frame_id": 2478,
  "material_class": "HDPE_Flake",
  "confidence": 0.9419,
  "bbox_xyxy": [313, 327, 622, 341],
  "inference_latency_ms": 11.51,
  "actuator_trigger_delay_ms": 388.49,
  "action": "AIR_BLAST_VALVE_1"
}
