import cv2
import mediapipe as mp
from flask import Flask, Response
from pathlib import Path
from urllib.request import urlretrieve
import time

class FaceAnalysis:
    """
    Pipeline:
    1) Face Detection (MediaPipe)
    2) Bounding box drawing (OpenCV)
    """

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        self._min_detection_confidence = min_detection_confidence
        self._face_detector = None
        self._mode = "unknown"
        self._init_detector()

    def _init_detector(self):
        # Try classic solutions API first.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self._mode = "solutions"
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self._min_detection_confidence,
            )
            return

        # Fallback for modern MediaPipe builds (e.g. Python 3.13).
        self._mode = "tasks"
        model_path = self._ensure_tasks_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self._min_detection_confidence,
        )
        self._face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _ensure_tasks_model() -> Path:
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "blaze_face_short_range.tflite"
        if model_path.exists():
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        )
        urlretrieve(model_url, model_path)
        return model_path

    @staticmethod
    def _relative_bbox_to_xyxy(relative_bbox, frame_width: int, frame_height: int):
        xmin = int(relative_bbox.xmin * frame_width)
        ymin = int(relative_bbox.ymin * frame_height)
        width = int(relative_bbox.width * frame_width)
        height = int(relative_bbox.height * frame_height)

        xmax = xmin + width
        ymax = ymin + height

        # Clamp to frame boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(frame_width - 1, xmax)
        ymax = min(frame_height - 1, ymax)

        return xmin, ymin, xmax, ymax

    def process_frame(self, frame):
        """
        Returns:
            annotated_frame: frame with drawn bounding boxes
            detections_data: list of tuples (xmin, ymin, xmax, ymax, confidence)
        """
        if self._mode == "solutions":
            return self._process_with_solutions(frame)
        return self._process_with_tasks(frame)

    def _process_with_solutions(self, frame):
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detector.process(rgb_frame)
        detections_data = []
        face_boxes = []

        if results.detections:
            for detection in results.detections:
                relative_bbox = detection.location_data.relative_bounding_box
                xmin, ymin, xmax, ymax = self._relative_bbox_to_xyxy(
                    relative_bbox, frame_width, frame_height
                )
                confidence = float(detection.score[0]) if detection.score else 0.0
                detections_data.append((xmin, ymin, xmax, ymax, confidence))
                face_boxes.append((xmin, ymin, xmax, ymax, confidence))

        annotated_frame = self._apply_black_background(frame, face_boxes)
        for xmin, ymin, xmax, ymax, confidence in face_boxes:
            self._draw_detection(annotated_frame, xmin, ymin, xmax, ymax, confidence)
        return annotated_frame, detections_data

    def _process_with_tasks(self, frame):
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._face_detector.detect(mp_image)

        detections_data = []
        face_boxes = []

        for detection in result.detections:
            bbox = detection.bounding_box
            xmin = max(0, int(bbox.origin_x))
            ymin = max(0, int(bbox.origin_y))
            xmax = min(frame_width - 1, int(bbox.origin_x + bbox.width))
            ymax = min(frame_height - 1, int(bbox.origin_y + bbox.height))
            confidence = (
                float(detection.categories[0].score)
                if detection.categories
                else 0.0
            )
            detections_data.append((xmin, ymin, xmax, ymax, confidence))
            face_boxes.append((xmin, ymin, xmax, ymax, confidence))

        annotated_frame = self._apply_black_background(frame, face_boxes)
        for xmin, ymin, xmax, ymax, confidence in face_boxes:
            self._draw_detection(annotated_frame, xmin, ymin, xmax, ymax, confidence)
        return annotated_frame, detections_data

    @staticmethod
    def _draw_detection(annotated_frame, xmin, ymin, xmax, ymax, confidence):
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"Face {confidence:.2f}"
        label_y = max(20, ymin - 10)
        cv2.putText(
            annotated_frame,
            label,
            (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _apply_black_background(frame, face_boxes):
        output = frame.copy()
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask[:] = 0

        for xmin, ymin, xmax, ymax, _ in face_boxes:
            mask[ymin:ymax, xmin:xmax] = 255

        return cv2.bitwise_and(output, output, mask=mask)

    def close(self):
        self._face_detector.close()


app = Flask(__name__)

cap = None
face_analysis = None
TARGET_FPS = 15


def _open_first_available_camera(max_index: int = 3):
    # On Windows, DirectShow often gives more stable camera access.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    for backend in backends:
        for idx in range(max_index + 1):
            candidate = cv2.VideoCapture(idx, backend)
            if candidate.isOpened():
                return candidate, idx
            candidate.release()
    return None, None


def _ensure_pipeline_initialized():
    global cap, face_analysis
    if face_analysis is None:
        face_analysis = FaceAnalysis(min_detection_confidence=0.5, model_selection=0)
    if cap is None or not cap.isOpened():
        cap, camera_idx = _open_first_available_camera(max_index=3)
        if cap is None:
            raise RuntimeError("Cannot open camera. Check camera permissions/device.")
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(f"Using camera index: {camera_idx}")


def generate_mjpeg_stream():
    _ensure_pipeline_initialized()
    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now < next_frame_time:
            time.sleep(next_frame_time - now)
        next_frame_time = max(next_frame_time + frame_interval, time.perf_counter())

        success, frame = cap.read()
        if not success:
            continue

        annotated_frame, detections = face_analysis.process_frame(frame)

        # Optional console output in requested format:
        # (xmin, ymin, xmax, ymax, confidence)
        if detections:
            print(detections)

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>FaceAnalysis Stream</title></head>"
        "<body style='margin:0; background:#000; color:#fff; font-family:Arial,sans-serif;'>"
        "<div style='min-height:100vh; display:flex; flex-direction:column; "
        "align-items:center; justify-content:center; gap:12px;'>"
        "<h2 style='margin:0;'>FaceAnalysis Stream</h2>"
        "<p style='margin:0;'>Open stream: <a style='color:#9ad1ff;' href='/video_feed'>/video_feed</a></p>"
        "<img src='/video_feed' width='960' style='max-width:95vw; border:1px solid #333;' />"
        "</div></body></html>"
    )


@app.route("/video_feed")
def video_feed():
    try:
        return Response(
            generate_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except RuntimeError as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        if face_analysis is not None:
            face_analysis.close()
        if cap is not None:
            cap.release()
