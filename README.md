# V.O.I.C.E
Visual Orientation Interactive Computing Entity

## `rate.py` — пайплайн FaceAnalysis (ТЗ)

1. **Детекція** — `FaceAnalysis`: MediaPipe Face Detection, bbox + confidence.
2. **Лендмарки та вирівнювання** — `FaceAlignment`: Face Landmarker (mesh + райдужки), `extract_keypoints`, `align_face`; на відео — контур/точки та прев’ю вирівняного обличчя.
3. **Поза голови** — `HeadPoseEstimator`: OpenCV `solvePnP` → Pitch / Yaw / Roll, осі на кадрі.

Додатково: **`is_watching`** (погляд у камеру за райдужками), **`head_ok`** (голова до камери за кутами).  
Програмний збір без Flask: `analyze_frame_full_pipeline(frame_bgr, face_analysis, face_alignment, head_pose)`.

Запуск стріму: `python rate.py` → `http://127.0.0.1:5000/stream`
