import cv2
import mediapipe as mp
from flask import Flask, Response, render_template_string
from pathlib import Path
from urllib.request import urlretrieve
import json
import os
import time
import sqlite3
import threading
from typing import Any, Dict, List, Tuple, Optional, Set, Union
import numpy as np
try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None
try:
    # InsightFace face embeddings
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
except Exception:  # pragma: no cover
    InsightFaceAnalysis = None


def _load_env_from_dotenv() -> None:
    """
    Мини-замена python-dotenv: читает переменные из ".env" рядом с файлом.
    Поддерживает формат: KEY=value (комментарии/пустые строки игнорируются).
    """
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return
    try:
        text = dotenv_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or not val:
            continue
        # Поддерживаем как OPENAI_API_KEY=..., так и api_key=...
        if key == "api_key" and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = val
        if key not in os.environ:
            os.environ[key] = val


_load_env_from_dotenv()

class FaceAnalysis:
    """
    Крок 1 ТЗ: Face Detection (MediaPipe Face Detection / BlazeFace short-range TFLite).
    Вихід: (xmin, ymin, xmax, ymax, confidence); рамки малюються OpenCV.
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

        annotated_frame = frame.copy()
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

        annotated_frame = frame.copy()
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

    def close(self):
        self._face_detector.close()


class FaceAlignment:
    """
    Крок 2 ТЗ: Face Alignment & Landmarks (MediaPipe Face Landmarker / Face Mesh).

    Outputs:
        - get_face_landmarks_xyz: масив (x, y, z) на кожне обличчя (x,y — пікселі).
        - extract_keypoints: іменовані точки (очі, ніс, рот, підборіддя) + контур jawline.
        - align_face: 2D affine warp у канонічну позу (для подальшого розпізнавання).
    """

    KEYPOINT_IDXS: Dict[str, int] = {
        "left_eye_outer": 33,
        "left_eye_inner": 133,
        "right_eye_outer": 263,
        "right_eye_inner": 362,
        "nose_tip": 1,
        "mouth_left": 61,
        "mouth_right": 291,
        "chin": 152,
    }

    JAWLINE_IDXS: List[int] = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152,
    ]

    FACE_LANDMARKER_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_faces: int = 2,
        min_face_detection_confidence: float = 0.5,
        include_iris: bool = False,
    ):
        self._include_iris = include_iris
        self._landmarker = self._create_landmarker(
            model_path=model_path,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
        )

    @staticmethod
    def _ensure_model(model_path: Path) -> Path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            return model_path
        urlretrieve(FaceAlignment.FACE_LANDMARKER_MODEL_URL, model_path)
        return model_path

    def _create_landmarker(
        self,
        model_path: Optional[Path],
        num_faces: int,
        min_face_detection_confidence: float,
    ):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

        model_path = self._ensure_model(model_path)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        base_options = BaseOptions(model_asset_path=str(model_path))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
        )

        return FaceLandmarker.create_from_options(options)

    @staticmethod
    def _frame_to_mp_image(frame_bgr) -> "mp.Image":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def get_face_landmarks_xyz(
        self,
        frame_bgr,
    ) -> List[List[Tuple[float, float, float]]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        mp_image = self._frame_to_mp_image(frame_bgr)
        result = self._landmarker.detect(mp_image)

        faces_landmarks: List[List[Tuple[float, float, float]]] = []

        for face_landmarks in result.face_landmarks:
            pts: List[Tuple[float, float, float]] = []
            for i, lm in enumerate(face_landmarks):
                if not self._include_iris and i >= 468:
                    break
                x_px = float(lm.x) * frame_w
                y_px = float(lm.y) * frame_h
                z_val = float(lm.z)
                pts.append((x_px, y_px, z_val))
            faces_landmarks.append(pts)

        return faces_landmarks

    def extract_keypoints(
        self,
        face_landmarks_xyz: List[Tuple[float, float, float]],
    ) -> Dict[str, Union[Tuple[float, float, float], List[Tuple[float, float, float]]]]:
        """
        Ключові точки ТЗ: очі (кути), ніс, куточки губ, підборіддя + полілінія контуру підборіддя/овалу.
        """
        def get_by_idx(idx: int) -> Tuple[float, float, float]:
            if idx < 0 or idx >= len(face_landmarks_xyz):
                raise IndexError(
                    f"Landmark idx {idx} out of range (len={len(face_landmarks_xyz)})."
                )
            return face_landmarks_xyz[idx]

        keypoints: Dict[str, Union[Tuple[float, float, float], List[Tuple[float, float, float]]]] = {}
        for name, idx in self.KEYPOINT_IDXS.items():
            keypoints[name] = get_by_idx(idx)

        jawline: List[Tuple[float, float, float]] = []
        for idx in self.JAWLINE_IDXS:
            if idx < len(face_landmarks_xyz):
                jawline.append(get_by_idx(idx))
        keypoints["jawline_contour"] = jawline
        return keypoints

    def align_face(
        self,
        frame_bgr: np.ndarray,
        face_landmarks_xyz: List[Tuple[float, float, float]],
        output_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вирівнювання обличчя перед розпізнаванням: affine за центрами очей + кінчик носа.
        """
        (out_w, out_h) = output_size
        left_eye_outer = face_landmarks_xyz[self.KEYPOINT_IDXS["left_eye_outer"]]
        left_eye_inner = face_landmarks_xyz[self.KEYPOINT_IDXS["left_eye_inner"]]
        right_eye_outer = face_landmarks_xyz[self.KEYPOINT_IDXS["right_eye_outer"]]
        right_eye_inner = face_landmarks_xyz[self.KEYPOINT_IDXS["right_eye_inner"]]
        nose_tip = face_landmarks_xyz[self.KEYPOINT_IDXS["nose_tip"]]

        left_eye_center = (
            (left_eye_outer[0] + left_eye_inner[0]) * 0.5,
            (left_eye_outer[1] + left_eye_inner[1]) * 0.5,
        )
        right_eye_center = (
            (right_eye_outer[0] + right_eye_inner[0]) * 0.5,
            (right_eye_outer[1] + right_eye_inner[1]) * 0.5,
        )
        nose_pt = (nose_tip[0], nose_tip[1])

        src = np.float32([left_eye_center, right_eye_center, nose_pt])
        dst = np.float32(
            [
                (out_w * 0.35, out_h * 0.35),
                (out_w * 0.65, out_h * 0.35),
                (out_w * 0.50, out_h * 0.60),
            ]
        )
        m_affine = cv2.getAffineTransform(src, dst)
        aligned = cv2.warpAffine(frame_bgr, m_affine, (out_w, out_h))
        return aligned, m_affine

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()


class HeadPoseEstimator:
    """
    Head pose (Pitch, Yaw, Roll) via OpenCV solvePnP.

    Uses 6 2D landmarks from MediaPipe Face Mesh and a canonical 3D face model
    (same point ordering as common OpenCV head-pose tutorials).
    """

    # Order: nose tip, chin, left eye outer, right eye outer, mouth left, mouth right
    LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)

    # 3D reference model (approximate mm, frontal neutral face)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -330.0, -65.0),  # chin
            (-225.0, 170.0, -135.0),  # left eye outer
            (225.0, 170.0, -135.0),  # right eye outer
            (-150.0, -150.0, -125.0),  # left mouth corner
            (150.0, -150.0, -125.0),  # right mouth corner
        ],
        dtype=np.float64,
    )

    def __init__(self, refine: bool = True):
        self._refine = refine

    @staticmethod
    def _camera_matrix(frame_shape: Tuple[int, ...]) -> np.ndarray:
        h, w = frame_shape[0], frame_shape[1]
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        return np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_euler_degrees(R: np.ndarray) -> Tuple[float, float, float]:
        """Pitch (X), Yaw (Y), Roll (Z) in degrees — OpenCV camera convention."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0
        return (
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)),
        )

    def estimate(
        self,
        landmarks_xy: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[float, float, float, np.ndarray, np.ndarray]]:
        """
        Returns:
            (pitch_deg, yaw_deg, roll_deg, rvec, tvec) or None if solvePnP fails.
        """
        max_idx = max(self.LANDMARK_INDICES)
        if len(landmarks_xy) <= max_idx:
            return None

        image_points = np.array(
            [[landmarks_xy[i][0], landmarks_xy[i][1]] for i in self.LANDMARK_INDICES],
            dtype=np.float64,
        )

        cam_matrix = self._camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        if self._refine:
            rvec, tvec = cv2.solvePnPRefineVVS(
                self.MODEL_POINTS_3D,
                image_points,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )

        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler_degrees(R)
        return pitch, yaw, roll, rvec, tvec

    def draw_pose_axes(
        self,
        frame_bgr: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        length: float = 200.0,
    ) -> None:
        cam_matrix = self._camera_matrix(frame_bgr.shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        axis_pts = np.float32(
            [
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length],
            ]
        )
        proj, _ = cv2.projectPoints(axis_pts, rvec, tvec, cam_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2).astype(int)
        origin = tuple(proj[0])
        cv2.line(frame_bgr, origin, tuple(proj[1]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(frame_bgr, origin, tuple(proj[2]), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(frame_bgr, origin, tuple(proj[3]), (255, 0, 0), 3, cv2.LINE_AA)


# Очі + райдужка (Face Landmarker, 478 точок): чи зіниці біля центру «вікна» ока.
# Якщо дивиться в камеру, райдужка ≈ по центру між зовнішнім/внутрішнім кутом і між верхом/низом повіки.
LEFT_IRIS_IDXS = (474, 475, 476, 477)
RIGHT_IRIS_IDXS = (469, 470, 471, 472)
# Ліве око: зовнішній кут, внутрішній; верх / низ (MediaPipe mesh)
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 263, 362
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

# Допустиме відхилення нормалізованого параметра від 0.5 (серединa сегмента ока)
EYE_GAZE_CENTER_TOLERANCE = 0.17


def _mean_iris_xy(
    lms: List[Tuple[float, float, float]], idxs: Tuple[int, ...]
) -> Tuple[float, float]:
    xs = [lms[i][0] for i in idxs]
    ys = [lms[i][1] for i in idxs]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _segment_param(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Проєкція точки (px,py) на відрізок A→B; 0 = A, 1 = B (без clamp)."""
    ex, ey = bx - ax, by - ay
    denom = ex * ex + ey * ey
    if denom < 1e-6:
        return 0.5
    return ((px - ax) * ex + (py - ay) * ey) / denom


def compute_is_watching_from_eyes(
    lms: List[Tuple[float, float, float]],
) -> bool:
    """
    True, якщо за положенням райдужок обидва ока «дивляться» приблизно в камеру
    (райдужка близько до центру ока по горизонталі та вертикалі).
    Потрібні всі 478 лендмарків (райдужка увімкнена в FaceAlignment).
    """
    need = max(
        max(LEFT_IRIS_IDXS),
        max(RIGHT_IRIS_IDXS),
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
    )
    if len(lms) <= need:
        return False

    lix, liy = _mean_iris_xy(lms, LEFT_IRIS_IDXS)
    rix, riy = _mean_iris_xy(lms, RIGHT_IRIS_IDXS)

    # Горизонталь: зовнішній → внутрішній кут
    t_h_l = _segment_param(
        lix, liy,
        lms[LEFT_EYE_OUTER][0], lms[LEFT_EYE_OUTER][1],
        lms[LEFT_EYE_INNER][0], lms[LEFT_EYE_INNER][1],
    )
    t_h_r = _segment_param(
        rix, riy,
        lms[RIGHT_EYE_OUTER][0], lms[RIGHT_EYE_OUTER][1],
        lms[RIGHT_EYE_INNER][0], lms[RIGHT_EYE_INNER][1],
    )
    # Вертикаль: верх повіки → низ
    t_v_l = _segment_param(
        lix, liy,
        lms[LEFT_EYE_TOP][0], lms[LEFT_EYE_TOP][1],
        lms[LEFT_EYE_BOTTOM][0], lms[LEFT_EYE_BOTTOM][1],
    )
    t_v_r = _segment_param(
        rix, riy,
        lms[RIGHT_EYE_TOP][0], lms[RIGHT_EYE_TOP][1],
        lms[RIGHT_EYE_BOTTOM][0], lms[RIGHT_EYE_BOTTOM][1],
    )

    th = EYE_GAZE_CENTER_TOLERANCE
    for t in (t_h_l, t_h_r, t_v_l, t_v_r):
        if abs(t - 0.5) > th:
            return False
    return True


def _match_landmarks_to_detection(
    lms: List[Tuple[float, float, float]],
    detections: List[Tuple],
    used_indices: Set[int],
) -> Optional[int]:
    """Індекс у `detections`, якому відповідає mesh (ніс у bbox; інакше найближчий центр)."""
    if not detections or len(lms) < 2:
        return None
    nx, ny = float(lms[1][0]), float(lms[1][1])
    inside: List[Tuple[float, int]] = []
    all_d: List[Tuple[float, int]] = []
    for i, det in enumerate(detections):
        if i in used_indices:
            continue
        xmin, ymin, xmax, ymax = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        d = (nx - cx) ** 2 + (ny - cy) ** 2
        all_d.append((d, i))
        if xmin <= nx <= xmax and ymin <= ny <= ymax:
            inside.append((d, i))
    pool = inside if inside else all_d
    if not pool:
        return None
    pool.sort(key=lambda x: x[0])
    return pool[0][1]


def draw_is_watching_next_to_face_label(
    frame_bgr: np.ndarray,
    xmin: int,
    ymin: int,
    confidence: float,
    is_watching: bool,
    user_name: Optional[str] = None,
) -> None:
    """Текст одразу після напису «Face 0.xx» (ті самі шрифт / масштаб, що в _draw_detection)."""
    face_label = f"Face {confidence:.2f}"
    label_y = max(20, ymin - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, _), _ = cv2.getTextSize(face_label, font, font_scale, thickness)
    if user_name:
        suffix = f"  {user_name} is_watching={is_watching}"
    else:
        suffix = f"  is_watching={is_watching}"
    watch_color = (0, 255, 0) if is_watching else (80, 80, 255)
    x_suffix = xmin + tw + 2
    cv2.putText(
        frame_bgr,
        suffix,
        (x_suffix, label_y),
        font,
        font_scale,
        watch_color,
        thickness,
        cv2.LINE_AA,
    )


# ----------------------------
# Простые "user embeddings"
# ----------------------------

EMBEDDING_DB_PATH = Path(__file__).resolve().parent / "embeddings.db"

# embedding считается как вектор по выровненному лицу.
# Важно: это не нейросетевой face embedding (в репозитории нет модели),
# а стабильный вектор-описатель, чтобы можно было различать лица по сходству.
# (w, h) должно оставаться неизменным, чтобы старые embeddings оставались сравнимыми.
EMBEDDING_OUTPUT_SIZE: Tuple[int, int] = (32, 32)  # (w, h)

# Как в ultv1.py: порог по dot product / cosine similarity.
DEFAULT_SIMILARITY_THRESHOLD = 0.45

# Доп. условие "четкости": лучший матч должен быть лучше второго.
# Чуть-чуть оставляем, чтобы уменьшить дрейф и лишние user_N.
SIMILARITY_MARGIN = 0.01

# В ultv1.py embedding при совпадении НЕ обновляется.
# Но оставляем константу, чтобы не менять сигнатуры/структуру кода.
EMBEDDING_EMA_MOMENTUM = 0.98

# Параметры для голосового embedding при регистрации нового пользователя.
VOICE_SAMPLE_SECONDS = 2.0
VOICE_SAMPLE_RATE = 16000
VOICE_EMBEDDING_DIM = 256

# InsightFace model init is heavy, so we create it lazily.
_insightface_app = None
_insightface_lock = threading.Lock()


def _compute_embedding_from_aligned(aligned_bgr: np.ndarray) -> np.ndarray:
    """Возвращает L2-нормализованный вектор float32 длины W*H."""
    target_dim = int(EMBEDDING_OUTPUT_SIZE[0] * EMBEDDING_OUTPUT_SIZE[1])

    # Предпочтительно: InsightFace embedding.
    if InsightFaceAnalysis is not None:
        global _insightface_app
        try:
            with _insightface_lock:
                if _insightface_app is None:
                    _insightface_app = InsightFaceAnalysis(
                        name="buffalo_s",
                        providers=["CPUExecutionProvider"],
                    )
                    _insightface_app.prepare(ctx_id=-1, det_size=(320, 320))

            # InsightFace любит, чтобы вход был не слишком маленьким.
            face_img = aligned_bgr
            min_side = min(face_img.shape[0], face_img.shape[1])
            if min_side < 96:
                face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_LINEAR)

            faces = _insightface_app.get(face_img)
            if faces:
                face = faces[0]
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(face, "embedding", None)
                if emb is not None:
                    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                    # Приводим к ожидаемой размерности, чтобы не менять БД.
                    if vec.size < target_dim:
                        padded = np.zeros((target_dim,), dtype=np.float32)
                        padded[: vec.size] = vec
                        vec = padded
                    else:
                        vec = vec[:target_dim]

                    # Убедимся, что вектор нормализован.
                    norm = float(np.linalg.norm(vec)) + 1e-12
                    vec /= norm
                    return vec
        except Exception as exc:
            # Если InsightFace не сработал (нет модели/ошибка ввода) — fallback ниже.
            print(f"[embedding] InsightFace failed, fallback. Reason: {exc}")

    # Fallback: старый пиксельный embedding (чтобы код не падал).
    w, h = EMBEDDING_OUTPUT_SIZE
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    resized_eq = clahe.apply(resized)
    vec = resized_eq.astype(np.float32).reshape(-1)
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec)) + 1e-12
    vec /= norm
    return vec


def _compute_voice_embedding(audio_data: np.ndarray) -> np.ndarray:
    """
    Возвращает L2-нормализованный voice embedding float32 фиксированной длины.
    Легковесный DSP-вектор: log power spectrum (без внешних ML-зависимостей).
    """
    audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Empty audio for voice embedding.")

    # Убираем DC и нормализуем амплитуду.
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio))) + 1e-12
    audio = audio / peak

    # Окно Хэнна + спектр мощности.
    window = np.hanning(audio.size).astype(np.float32)
    spectrum = np.fft.rfft(audio * window)
    power = np.abs(spectrum).astype(np.float32)

    # Сжимаем/растягиваем до фиксированной размерности для хранения в БД.
    x_old = np.linspace(0.0, 1.0, num=power.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=VOICE_EMBEDDING_DIM, dtype=np.float32)
    vec = np.interp(x_new, x_old, power).astype(np.float32)
    vec = np.log1p(vec)

    norm = float(np.linalg.norm(vec)) + 1e-12
    vec /= norm
    return vec


def _record_voice_embedding() -> Optional[np.ndarray]:
    """
    Пишет короткий фрагмент с микрофона и возвращает embedding голоса.
    Если микрофон/библиотека недоступны — None.
    """
    if sd is None:
        print("[voice] sounddevice is not available, skip voice registration.")
        return None
    try:
        num_samples = int(VOICE_SAMPLE_SECONDS * VOICE_SAMPLE_RATE)
        rec = sd.rec(
            num_samples,
            samplerate=VOICE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        return _compute_voice_embedding(rec)
    except Exception as exc:
        print(f"[voice] failed to record/encode voice: {exc}")
        return None


class EmbeddingsDB:
    """Хранит embedding’и пользователей в SQLite и выдаёт user_1, user_2, ..."""

    def __init__(
        self,
        db_path: Path,
        embedding_dim: int,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)

        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    user_name TEXT PRIMARY KEY,
                    face_embedding BLOB NOT NULL,
                    voice_embedding BLOB,
                    created_at REAL NOT NULL
                )
                """
            )
            self._ensure_schema_compatibility()

        # В памяти держим embedding на пользователя.
        self._known: Dict[str, np.ndarray] = {}
        self._next_user_id = 1
        self._load_known_embeddings()

    @staticmethod
    def _serialize_embedding(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size != self._embedding_dim:
            raise ValueError(
                f"Unexpected embedding dim: {vec.size} (expected {self._embedding_dim})"
            )
        return vec

    def _ensure_schema_compatibility(self) -> None:
        """Мягкая миграция старой схемы (embedding -> face_embedding)."""
        cols = self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        col_names = {row[1] for row in cols}

        if "face_embedding" not in col_names:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN face_embedding BLOB")
            if "embedding" in col_names:
                self._conn.execute(
                    "UPDATE embeddings SET face_embedding = embedding WHERE face_embedding IS NULL"
                )

        if "voice_embedding" not in col_names:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN voice_embedding BLOB")

    def _load_known_embeddings(self) -> None:
        with self._lock, self._conn:
            cols = self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
            col_names = {row[1] for row in cols}
            if "face_embedding" in col_names:
                rows = self._conn.execute(
                    "SELECT user_name, face_embedding FROM embeddings"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT user_name, embedding FROM embeddings"
                ).fetchall()

        known: Dict[str, np.ndarray] = {}
        for user_name, blob in rows:
            try:
                known[user_name] = self._deserialize_embedding(blob)
            except Exception:
                continue

        self._known = known

        max_id = 0
        for name in self._known.keys():
            if not name.startswith("user_"):
                continue
            suffix = name[len("user_") :]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
        self._next_user_id = max_id + 1

    def _get_best_two_matches(
        self, embedding_vec: np.ndarray
    ) -> Tuple[Optional[str], float, Optional[str], float]:
        best_name: Optional[str] = None
        best_sim: float = -1.0
        second_name: Optional[str] = None
        second_sim: float = -1.0

        for name, known_vec in self._known.items():
            # embedding_vec и known_vec уже L2-нормализованы => dot = cosine similarity
            sim = float(np.dot(embedding_vec, known_vec))
            if sim > best_sim:
                second_name, second_sim = best_name, best_sim
                best_name, best_sim = name, sim
            elif sim > second_sim:
                second_name, second_sim = name, sim

        return best_name, best_sim, second_name, second_sim

    def get_or_create_user_name(
        self, embedding_vec: np.ndarray, voice_embedding_vec: Optional[np.ndarray] = None
    ) -> Tuple[str, bool]:
        with self._lock:
            if not self._known:
                user_name = f"user_{self._next_user_id}"
                self._next_user_id += 1
                self._known[user_name] = embedding_vec
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO embeddings (user_name, face_embedding, voice_embedding, created_at) VALUES (?, ?, ?, ?)",
                        (
                            user_name,
                            self._serialize_embedding(embedding_vec),
                            self._serialize_embedding(voice_embedding_vec)
                            if voice_embedding_vec is not None
                            else None,
                            time.time(),
                        ),
                    )
                return user_name, True

            best_name, best_sim, second_name, second_sim = self._get_best_two_matches(embedding_vec)
            # Как в ultv1.py: достаточно лучшего совпадения выше порога.
            if best_name is not None and best_sim >= self._similarity_threshold:
                return best_name, False

            user_name = f"user_{self._next_user_id}"
            self._next_user_id += 1
            self._known[user_name] = embedding_vec
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (user_name, face_embedding, voice_embedding, created_at) VALUES (?, ?, ?, ?)",
                    (
                        user_name,
                        self._serialize_embedding(embedding_vec),
                        self._serialize_embedding(voice_embedding_vec)
                        if voice_embedding_vec is not None
                        else None,
                        time.time(),
                    ),
                )
            return user_name, True

    def set_voice_embedding(self, user_name: str, voice_embedding_vec: np.ndarray) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE embeddings SET voice_embedding = ? WHERE user_name = ?",
                (self._serialize_embedding(voice_embedding_vec), user_name),
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# Кольори для накладання кроку 2 (різні обличчя)
_STEP2_FACE_COLORS: List[Tuple[int, int, int]] = [
    (0, 200, 255),
    (255, 150, 80),
    (180, 100, 255),
]


def draw_step2_landmarks_overlay(
    frame_bgr: np.ndarray,
    lms: List[Tuple[float, float, float]],
    face_index: int,
) -> None:
    """Малює контур підборіддя/овалу та ключові точки (очі, ніс, рот, підборіддя)."""
    if len(lms) < 152:
        return
    color = _STEP2_FACE_COLORS[face_index % len(_STEP2_FACE_COLORS)]
    jaw_pts = []
    for idx in FaceAlignment.JAWLINE_IDXS:
        if idx < len(lms):
            jaw_pts.append([int(lms[idx][0]), int(lms[idx][1])])
    if len(jaw_pts) >= 2:
        arr = np.array(jaw_pts, dtype=np.int32)
        cv2.polylines(frame_bgr, [arr], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    for idx in FaceAlignment.KEYPOINT_IDXS.values():
        if idx < len(lms):
            x, y = int(lms[idx][0]), int(lms[idx][1])
            cv2.circle(frame_bgr, (x, y), 3, color, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def paste_aligned_face_thumbnail(
    frame_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
    margin: int = 10,
) -> None:
    """Вставляє вирівняне обличчя у правий верхній кут (прев’ю кроку 2)."""
    fh, fw = frame_bgr.shape[:2]
    th, tw = aligned_bgr.shape[:2]
    x1 = max(margin, fw - tw - margin)
    y1 = margin
    y2, x2 = y1 + th, x1 + tw
    if y2 > fh - margin or x2 > fw - margin:
        return
    frame_bgr[y1:y2, x1:x2] = aligned_bgr
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        "Aligned (step 2)",
        (x1, max(y1 - 4, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def is_facing_camera_from_head_pose(
    pitch: float,
    yaw: float,
    roll: float,
    max_pitch: float = 22.0,
    max_yaw: float = 22.0,
    max_roll: float = 35.0,
) -> bool:
    """
    Додатково до погляду очима (is_watching): чи голова повернута до камери за кутами SolvePnP (крок 3 ТЗ).
    """
    return (
        abs(pitch) <= max_pitch
        and abs(yaw) <= max_yaw
        and abs(roll) <= max_roll
    )


def analyze_frame_full_pipeline(
    frame_bgr: np.ndarray,
    fa: FaceAnalysis,
    fal: FaceAlignment,
    hp: HeadPoseEstimator,
) -> Dict[str, Any]:
    """
    Повний послідовний пайплайн ТЗ (1→2→3) для одного кадру.
    Повертає структуровані дані без Flask.
    """
    annotated, detections = fa.process_frame(frame_bgr)
    faces_lms = fal.get_face_landmarks_xyz(frame_bgr)
    faces_info: List[Dict[str, Any]] = []

    for fi, lms in enumerate(faces_lms):
        try:
            keypoints = fal.extract_keypoints(lms)
        except IndexError:
            keypoints = None
        est = hp.estimate(lms, frame_bgr.shape)
        pose = None
        facing_head = None
        if est is not None:
            pitch, yaw, roll, _, _ = est
            pose = {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll}
            facing_head = is_facing_camera_from_head_pose(pitch, yaw, roll)

        try:
            aligned, affine = fal.align_face(frame_bgr, lms, output_size=(256, 256))
        except Exception:
            aligned, affine = None, None

        faces_info.append(
            {
                "face_index": fi,
                "landmarks_xyz": lms,
                "keypoints": keypoints,
                "head_pose_deg": pose,
                "is_facing_camera_head": facing_head,
                "is_watching_eyes": compute_is_watching_from_eyes(lms),
                "aligned_face_bgr": aligned,
                "affine_2x3": affine,
            }
        )

    return {
        "annotated_frame": annotated,
        "detections": detections,
        "faces": faces_info,
    }


app = Flask(__name__)

cap = None
face_analysis = None
face_alignment = None
head_pose = HeadPoseEstimator(refine=True)
TARGET_FPS = 15
embeddings_db: Optional[EmbeddingsDB] = None

# Shared camera: один worker пишет кадры и JSON-снимок; MJPEG-генераторы только читают.
_frame_lock = threading.Lock()
_latest_raw_frame: Optional[np.ndarray] = None
_latest_overlay_frame: Optional[np.ndarray] = None
_vision_snapshot: Dict[str, Any] = {}
_camera_worker_started = False

OPENAI_REALTIME_MODEL = os.environ.get(
    "OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"
)


def _serialize_keypoints_for_api(
    kp: Optional[Dict[str, Union[Tuple[float, float, float], List[Tuple[float, float, float]]]]],
) -> Optional[Dict[str, Any]]:
    if not kp:
        return None
    out: Dict[str, Any] = {}
    for name, val in kp.items():
        if name == "jawline_contour" and isinstance(val, list):
            out[name] = {"num_points": len(val)}
        elif isinstance(val, tuple) and len(val) == 3:
            out[name] = [
                round(float(val[0]), 2),
                round(float(val[1]), 2),
                round(float(val[2]), 2),
            ]
        elif isinstance(val, list):
            out[name] = {"num_points": len(val)}
        else:
            out[name] = str(val)
    return out


def _start_camera_worker() -> None:
    global _camera_worker_started
    with _frame_lock:
        if _camera_worker_started:
            return
        _camera_worker_started = True
    t = threading.Thread(target=_camera_worker_loop, daemon=True)
    t.start()


def _camera_worker_loop() -> None:
    global _latest_raw_frame, _latest_overlay_frame, _vision_snapshot
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

        raw_copy = frame.copy()
        annotated_frame, detections = face_analysis.process_frame(frame)
        faces_snapshots: List[Dict[str, Any]] = []

        try:
            faces_lms = face_alignment.get_face_landmarks_xyz(frame)
            used_detection_indices: Set[int] = set()
            for fi, lms in enumerate(faces_lms):
                try:
                    keypoints = face_alignment.extract_keypoints(lms)
                except IndexError:
                    keypoints = None

                draw_step2_landmarks_overlay(annotated_frame, lms, fi)

                if fi == 0:
                    try:
                        aligned_thumb, _ = face_alignment.align_face(
                            frame, lms, output_size=(112, 112)
                        )
                        paste_aligned_face_thumbnail(annotated_frame, aligned_thumb)
                    except (IndexError, cv2.error):
                        pass

                assigned_user_name: Optional[str] = None
                try:
                    aligned_for_embedding, _ = face_alignment.align_face(
                        frame,
                        lms,
                        output_size=EMBEDDING_OUTPUT_SIZE,
                    )
                    embedding_vec = _compute_embedding_from_aligned(aligned_for_embedding)
                    assert embeddings_db is not None
                    assigned_user_name, is_new_user = embeddings_db.get_or_create_user_name(
                        embedding_vec
                    )
                    if is_new_user:
                        voice_embedding_vec = _record_voice_embedding()
                        if voice_embedding_vec is not None:
                            embeddings_db.set_voice_embedding(
                                assigned_user_name, voice_embedding_vec
                            )
                            print(f"[voice] saved for {assigned_user_name}")
                        print(f"[embeddings] created {assigned_user_name}")
                except Exception as exc:
                    print(f"[embeddings] failed for face {fi}: {exc}")

                est = head_pose.estimate(lms, frame.shape)
                is_watching = compute_is_watching_from_eyes(lms)
                facing_head: Optional[bool] = None
                pitch = yaw = roll = None

                if est is None:
                    if detections:
                        di = _match_landmarks_to_detection(
                            lms, detections, used_detection_indices
                        )
                        if di is not None:
                            used_detection_indices.add(di)
                            xmin, ymin, xmax, ymax, conf = detections[di]
                            draw_is_watching_next_to_face_label(
                                annotated_frame,
                                int(xmin),
                                int(ymin),
                                float(conf),
                                is_watching,
                                user_name=assigned_user_name,
                            )
                    faces_snapshots.append(
                        {
                            "face_index": fi,
                            "user": assigned_user_name,
                            "landmarks_count": len(lms),
                            "face_landmarker_keypoints": _serialize_keypoints_for_api(
                                keypoints
                            ),
                            "is_watching_eyes": is_watching,
                            "head_pose_deg": None,
                            "head_ok": None,
                        }
                    )
                    continue

                pitch, yaw, roll, rvec, tvec = est
                facing_head = is_facing_camera_from_head_pose(pitch, yaw, roll)

                if detections:
                    di = _match_landmarks_to_detection(
                        lms, detections, used_detection_indices
                    )
                    if di is not None:
                        used_detection_indices.add(di)
                        xmin, ymin, xmax, ymax, conf = detections[di]
                        draw_is_watching_next_to_face_label(
                            annotated_frame,
                            int(xmin),
                            int(ymin),
                            float(conf),
                            is_watching,
                            user_name=assigned_user_name,
                        )

                head_pose.draw_pose_axes(annotated_frame, rvec, tvec, length=120.0)
                label = (
                    f"{assigned_user_name or 'unknown'} "
                    f"P:{pitch:+.0f} Y:{yaw:+.0f} R:{roll:+.0f} "
                    f"eyes:{is_watching} head_ok:{facing_head}"
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (10, 30 + fi * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                faces_snapshots.append(
                    {
                        "face_index": fi,
                        "user": assigned_user_name,
                        "landmarks_count": len(lms),
                        "face_landmarker_keypoints": _serialize_keypoints_for_api(
                            keypoints
                        ),
                        "is_watching_eyes": is_watching,
                        "head_pose_deg": {
                            "pitch_deg": float(pitch),
                            "yaw_deg": float(yaw),
                            "roll_deg": float(roll),
                        },
                        "head_ok": facing_head,
                    }
                )
        except Exception as exc:
            print(f"[head_pose] {exc}")

        snap = {
            "ts": time.time(),
            "detections": [
                [int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4])]
                for d in (detections or [])
            ],
            "faces": faces_snapshots,
        }

        with _frame_lock:
            _latest_raw_frame = raw_copy
            _latest_overlay_frame = annotated_frame
            _vision_snapshot = snap


def _generate_mjpeg_from_buffer(overlay: bool):
    _start_camera_worker()
    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now < next_frame_time:
            time.sleep(next_frame_time - now)
        next_frame_time = max(next_frame_time + frame_interval, time.perf_counter())

        with _frame_lock:
            src = _latest_overlay_frame if overlay else _latest_raw_frame
            if src is None:
                continue
            frame = src.copy()

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


def generate_clean_mjpeg_stream():
    return _generate_mjpeg_from_buffer(overlay=False)


def generate_overlay_mjpeg_stream():
    return _generate_mjpeg_from_buffer(overlay=True)


def generate_mjpeg_stream():
    """Совместимость: прежнее поведение = оверлей."""
    return generate_overlay_mjpeg_stream()


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
    global cap, face_analysis, face_alignment, embeddings_db
    if face_analysis is None:
        face_analysis = FaceAnalysis(min_detection_confidence=0.5, model_selection=0)
    if face_alignment is None:
        face_alignment = FaceAlignment(
            num_faces=2,
            min_face_detection_confidence=0.5,
            include_iris=True,  # райдужки для is_watching по очах
        )
    if embeddings_db is None:
        w, h = EMBEDDING_OUTPUT_SIZE
        embedding_dim = w * h
        embeddings_db = EmbeddingsDB(
            db_path=EMBEDDING_DB_PATH,
            embedding_dim=embedding_dim,
            similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        )
    if cap is None or not cap.isOpened():
        cap, camera_idx = _open_first_available_camera(max_index=3)
        if cap is None:
            raise RuntimeError("Cannot open camera. Check camera permissions/device.")
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(f"Using camera index: {camera_idx}")


INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>V.O.I.C.E — камера + Realtime</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#0d0f12; color:#e8eaed; font-family:Segoe UI, Roboto, Arial, sans-serif; min-height:100vh; }
  .wrap { display:flex; flex-wrap:wrap; min-height:100vh; max-width:1400px; margin:0 auto; padding:12px; gap:16px; }
  .left { flex:1 1 420px; display:flex; flex-direction:column; gap:10px; }
  .right { flex:1 1 360px; min-width:280px; display:flex; flex-direction:column; gap:8px; }
  h1 { font-size:1.1rem; margin:0 0 4px 0; font-weight:600; color:#9ad1ff; }
  .video-box { background:#000; border:1px solid #2a3340; border-radius:8px; overflow:hidden; }
  .video-box img { display:block; width:100%; height:auto; vertical-align:middle; }
  button.toggle { align-self:flex-start; padding:10px 16px; border-radius:8px; border:1px solid #3d4a5c; background:#1a2332; color:#e8eaed; cursor:pointer; font-size:0.95rem; }
  button.toggle:hover { background:#243044; }
button.toggle:disabled { opacity: 0.55; cursor: not-allowed; }
  .details { display:none; flex-direction:column; gap:8px; }
  .details.visible { display:flex; }
  .details pre { margin:0; padding:10px; background:#11161d; border:1px solid #2a3340; border-radius:8px; font-size:11px; line-height:1.35; max-height:220px; overflow:auto; white-space:pre-wrap; word-break:break-word; }
  .details .ov { margin-top:4px; }
  .chat-panel { flex:1; display:flex; flex-direction:column; min-height:320px; background:#11161d; border:1px solid #2a3340; border-radius:8px; overflow:hidden; }
  .chat-panel h2 { margin:0; padding:10px 12px; font-size:0.95rem; border-bottom:1px solid #2a3340; background:#151b24; }
  #chatLog { flex:1; overflow-y:auto; padding:12px; font-size:0.9rem; line-height:1.45; }
  .msg-user { color:#7dd3fc; margin:6px 0; }
  .msg-agent { color:#a7f3d0; margin:6px 0; }
  .msg-sys { color:#94a3b8; font-size:0.85rem; margin:6px 0; }
  .hint { font-size:0.8rem; color:#64748b; padding:0 4px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <h1>Камера</h1>
    <div class="video-box">
      <img src="/stream" alt="video"/>
    </div>
    <button type="button" class="toggle" id="btnDetails">Показать детали анализа</button>
    <p class="hint" id="wsHint"></p>
    <div class="details" id="detailsPanel">
      <pre id="visionJson">Загрузка…</pre>
      <div class="video-box ov">
        <img id="overlayImg" src="/stream_overlay" alt="overlay"/>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="chat-panel">
      <h2>Диалог (OpenAI Realtime)</h2>
      <div id="chatLog"></div>
    </div>
    <button type="button" class="toggle" id="btnMic">Разрешить микрофон и начать разговор</button>
    <p class="hint">Микрофон: разрешите доступ в браузере. Нужны переменные окружения OPENAI_API_KEY на сервере и пакеты flask-sock, websocket-client.</p>
  </div>
</div>
<script>
(function(){
  const details = document.getElementById('detailsPanel');
  const btn = document.getElementById('btnDetails');
  const btnMic = document.getElementById('btnMic');
  const visionJson = document.getElementById('visionJson');
  const chatLog = document.getElementById('chatLog');
  const wsHint = document.getElementById('wsHint');
  let pollId = null;
  let detailsOn = false;
  let micStarted = false;
  let micPending = false;

  function logLine(cls, text) {
    const d = document.createElement('div');
    d.className = cls;
    d.textContent = text;
    chatLog.appendChild(d);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  btn.addEventListener('click', function() {
    detailsOn = !detailsOn;
    details.classList.toggle('visible', detailsOn);
    btn.textContent = detailsOn ? 'Скрыть детали анализа' : 'Показать детали анализа';
    if (detailsOn) {
      pollId = setInterval(function() {
        fetch('/api/vision_state').then(function(r){ return r.json(); }).then(function(j){
          visionJson.textContent = JSON.stringify(j, null, 2);
        }).catch(function(){ visionJson.textContent = 'Ошибка /api/vision_state'; });
      }, 250);
    } else {
      if (pollId) clearInterval(pollId);
      pollId = null;
    }
  });

  if (btnMic) {
    btnMic.addEventListener('click', function() {
      const isRealtimeOpen = !!realtime && realtime.readyState === WebSocket.OPEN;
      micPending = !isRealtimeOpen;
      wsHint.textContent = isRealtimeOpen
        ? 'Микрофон: запрошен'
        : 'Микрофон: запрошен, ждём подключение Realtime...';
      startMic();
      btnMic.disabled = true;
    });
  }

  const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = wsProto + '//' + location.host + '/ws/realtime';
  let realtime = null;

  function appendAgentDelta(t) {
    let last = chatLog.querySelector('.msg-agent.last');
    if (!last) {
      last = document.createElement('div');
      last.className = 'msg-agent last';
      chatLog.appendChild(last);
    }
    last.textContent = (last.textContent || '') + (t || '');
    chatLog.scrollTop = chatLog.scrollHeight;
  }
  function finishAgentTurn() {
    const last = chatLog.querySelector('.msg-agent.last');
    if (last) last.classList.remove('last');
  }

  function connectRealtime() {
    try {
      realtime = new WebSocket(wsUrl);
    } catch (e) {
      wsHint.textContent = 'WebSocket недоступен: ' + e;
      return;
    }
    realtime.addEventListener('open', function() {
      wsHint.textContent = 'Realtime: подключено';
      realtime.send(JSON.stringify({
        type: 'session.update',
        session: {
          modalities: ['text', 'audio'],
          instructions: 'Ты дружелюбный ассистент. Отвечай кратко по-русски, если пользователь говорит по-русски.',
          voice: 'alloy',
          input_audio_format: 'pcm16',
          output_audio_format: 'pcm16',
          turn_detection: { type: 'server_vad', create_response: true }
        }
      }));
      if (micPending) {
        startMic();
        if (btnMic) btnMic.disabled = true;
        micPending = false;
      }
    });
    realtime.addEventListener('message', function(ev) {
      let data = ev.data;
      if (typeof data !== 'string') return;
      let o;
      try { o = JSON.parse(data); } catch (e) { return; }
      const t = o.type || '';
      if (t === 'response.audio_transcript.delta' && o.delta) {
        appendAgentDelta(o.delta);
      } else if (t === 'response.text.delta' && o.delta) {
        appendAgentDelta(o.delta);
      } else if (t === 'response.audio_transcript.done' || t === 'response.done') {
        finishAgentTurn();
      } else if (t === 'conversation.item.input_audio_transcription.completed' && o.transcript) {
        logLine('msg-user', 'Вы: ' + o.transcript);
      } else if (t === 'error') {
        logLine('msg-sys', 'Ошибка: ' + JSON.stringify(o.error || o));
      }
    });
    realtime.addEventListener('close', function() {
      wsHint.textContent = 'Realtime: соединение закрыто';
    });
    realtime.addEventListener('error', function() {
      wsHint.textContent = 'Realtime: ошибка WebSocket (проверьте OPENAI_API_KEY и pip install flask-sock websocket-client)';
    });
  }

  function floatTo16BitPCM(float32Array) {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Array.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }

  function b64(buf) {
    let binary = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function startMic() {
    if (micStarted) return;
    micStarted = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      logLine('msg-sys', 'getUserMedia не поддерживается');
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC({ sampleRate: 24000 });
      const source = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      proc.onaudioprocess = function(e) {
        if (!realtime || realtime.readyState !== WebSocket.OPEN) return;
        let input = e.inputBuffer.getChannelData(0);
        if (ctx.sampleRate === 48000) {
          const out = new Float32Array(Math.floor(input.length / 2));
          for (let i = 0; i < out.length; i++) out[i] = input[i * 2];
          input = out;
        }
        const pcm = floatTo16BitPCM(input);
        realtime.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: b64(pcm)
        }));
      };
      source.connect(proc);
      proc.connect(mute);
      mute.connect(ctx.destination);
    }).catch(function(err) {
      logLine('msg-sys', 'Микрофон: ' + err);
    });
  }

  connectRealtime();
})();
</script>
</body>
</html>
"""


#FLASK PORT VIEW
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/vision_state")
def api_vision_state():
    with _frame_lock:
        snap = dict(_vision_snapshot)
    return Response(json.dumps(snap, ensure_ascii=False), mimetype="application/json; charset=utf-8")


@app.route("/stream")
def stream_clean():
    try:
        return Response(
            generate_clean_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except RuntimeError as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


@app.route("/stream_overlay")
def stream_overlay():
    try:
        return Response(
            generate_overlay_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except RuntimeError as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


@app.route("/stream_legacy")
def stream_legacy():
    """Прежний поток с оверлеем (то же, что /stream_overlay)."""
    try:
        return Response(
            generate_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except RuntimeError as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


try:
    from flask_sock import Sock

    sock = Sock(app)

    @sock.route("/ws/realtime")
    def realtime_ws(ws):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            try:
                dotenv_hint = str((Path(__file__).resolve().parent / ".env").resolve())
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "message": "OPENAI_API_KEY is not set (also checked .env).",
                                "dotenv_hint": dotenv_hint,
                                "expected_format": "OPENAI_API_KEY=your_key_here",
                            },
                        }
                    )
                )
            except Exception:
                pass
            return
        try:
            import websocket as ws_mod
        except ImportError:
            try:
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {"message": "pip install websocket-client"},
                        }
                    )
                )
            except Exception:
                pass
            return

        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        holder: List[Any] = [None]

        def on_open(ws_app):
            holder[0] = ws_app

        def on_message(_ws_app, message):
            try:
                if isinstance(message, bytes):
                    ws.send(message.decode("utf-8", errors="replace"))
                else:
                    ws.send(message)
            except Exception:
                pass

        def on_error(_ws_app, err):
            try:
                ws.send(
                    json.dumps(
                        {"type": "error", "error": {"message": str(err)}},
                    )
                )
            except Exception:
                pass

        def run_openai():
            ws_app = ws_mod.WebSocketApp(
                url,
                header=[
                    f"Authorization: Bearer {api_key}",
                    "OpenAI-Beta: realtime=v1",
                ],
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
            )
            ws_app.run_forever()

        t = threading.Thread(target=run_openai, daemon=True)
        t.start()
        deadline = time.perf_counter() + 15.0
        while holder[0] is None and time.perf_counter() < deadline:
            time.sleep(0.02)
        oa = holder[0]
        if oa is None:
            try:
                ws.send(json.dumps({"type": "error", "error": {"message": "OpenAI WS timeout"}}))
            except Exception:
                pass
            return
        try:
            while True:
                data = ws.receive()
                if data is None:
                    break
                oa.send(data)
        except Exception:
            pass
        try:
            oa.close()
        except Exception:
            pass

except ImportError:
    sock = None


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        if face_analysis is not None:
            face_analysis.close()
        if face_alignment is not None:
            face_alignment.close()
        if embeddings_db is not None:
            embeddings_db.close()
        if cap is not None:
            cap.release()
