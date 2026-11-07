import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

class ThreadedCamera:
    """Non-blocking camera ingestion thread for macOS."""

    def __init__(self, src: int | str = 0, target_fps: int = 30):
        self.src = src
        # On macOS, src=0 is usually the built-in FaceTime camera. 
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = False
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.target_fps = target_fps
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCamera":
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        frame_interval = 1.0 / self.target_fps
        synthetic_counter = 0

        while self.running:
            t0 = time.perf_counter()
            ret, frame = self.cap.read()

            if not ret:
                # Fallback: Generate a high-contrast mock plastic conveyor belt frame
                synthetic_counter += 1
                frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
                cv2.line(frame, (0, 360), (1280, 360), (60, 60, 60), 2)
                x_pos = (synthetic_counter * 18) % 1200 + 40
                cv2.ellipse(frame, (x_pos, 360), (45, 95), 25, 0, 360, (210, 210, 210), -1)
                cv2.putText(frame, "SIMULATED_FEED_MAC", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            with self.lock:
                self.frame = frame
                self.frame_id += 1

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def read_latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return 0, None
            return self.frame_id, self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()