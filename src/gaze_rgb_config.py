from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "last_aria_4.pt"
CROP_SIZE = 200
RESIZE_SIZE = 1080
DEFAULT_GAZE_PEAK_WINDOW = 3
DEFAULT_GAZE_PEAK_RADIUS = 15.0
DEFAULT_GAZE_BOUNDARY_RADIUS = 15.0
DEFAULT_GAZE_MIN_BOUNDARY_POINTS = 2
DEFAULT_GAZE_PEAK_YOLO_MAX_DISTANCE = 100.0

# Seconds /gaze_label keeps publishing after releasing b button
GAZE_LABEL_TAIL_SEC = 2.0
# time to wait for /transcription publish after /gaze_label is published
TRANSCRIPTION_HOLD_SEC = 0.5
