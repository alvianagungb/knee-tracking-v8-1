import cv2
import mediapipe as mp
import csv
import math
import statistics

from pathlib import Path
from datetime import datetime
from collections import deque


# ============================================================
# 1. KONFIGURASI KAMERA - DIPERTAHANKAN DARI V8
# ============================================================

CAMERA_INDEX = 1

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MIRROR_FRAME = False

MODEL_COMPLEXITY = 2

MIN_DETECTION_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60

VISIBILITY_THRESHOLD = 0.45


# ============================================================
# 2. KONFIGURASI OTOMATIS BASELINE V8.1
# ============================================================

CALIBRATION_FRAMES = 60

MAX_CALIBRATION_STD = 5.0

baseline_right = None
baseline_left = None

calibration_buffer = deque(
    maxlen=CALIBRATION_FRAMES
)

calibration_status = "SEARCHING"

calibration_message = "Mencari posisi awal yang stabil..."


# ============================================================
# 3. KONFIGURASI CSV
# ============================================================

OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SESSION_TIME = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

CSV_PATH = OUTPUT_DIR / f"knee_v8_1_{SESSION_TIME}.csv"


# ============================================================
# 4. INISIALISASI MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


# ============================================================
# 5. ENAM LANDMARK YANG DIPANTAU
# ============================================================

LANDMARK_IDS = {

    "RIGHT_HIP":
        mp_pose.PoseLandmark.RIGHT_HIP.value,

    "RIGHT_KNEE":
        mp_pose.PoseLandmark.RIGHT_KNEE.value,

    "RIGHT_ANKLE":
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,

    "LEFT_HIP":
        mp_pose.PoseLandmark.LEFT_HIP.value,

    "LEFT_KNEE":
        mp_pose.PoseLandmark.LEFT_KNEE.value,

    "LEFT_ANKLE":
        mp_pose.PoseLandmark.LEFT_ANKLE.value

}


# ============================================================
# 6. WARNA VISUALISASI
# ============================================================

RIGHT_COLOR = (0, 255, 0)

LEFT_COLOR = (255, 150, 0)

TEXT_COLOR = (255, 255, 255)


# ============================================================
# 7. FUNGSI KOORDINAT
# ============================================================

def landmark_point(landmark, width, height):

    return (

        int(landmark.x * width),

        int(landmark.y * height)

    )


# ============================================================
# 8. FUNGSI MENGHITUNG SUDUT LUTUT
# ============================================================

def calculate_knee_angle(
    hip,
    knee,
    ankle,
    width,
    height
):

    if (

        hip.visibility < VISIBILITY_THRESHOLD

        or knee.visibility < VISIBILITY_THRESHOLD

        or ankle.visibility < VISIBILITY_THRESHOLD

    ):

        return None

    hip_x, hip_y = landmark_point(
        hip,
        width,
        height
    )

    knee_x, knee_y = landmark_point(
        knee,
        width,
        height
    )

    ankle_x, ankle_y = landmark_point(
        ankle,
        width,
        height
    )

    vector_hip = (

        hip_x - knee_x,

        hip_y - knee_y

    )

    vector_ankle = (

        ankle_x - knee_x,

        ankle_y - knee_y

    )

    dot_product = (

        vector_hip[0] * vector_ankle[0]

        + vector_hip[1] * vector_ankle[1]

    )

    magnitude_hip = math.sqrt(

        vector_hip[0] ** 2

        + vector_hip[1] ** 2

    )

    magnitude_ankle = math.sqrt(

        vector_ankle[0] ** 2

        + vector_ankle[1] ** 2

    )

    denominator = (
        magnitude_hip * magnitude_ankle
    )

    if denominator == 0:

        return None

    cosine_angle = (
        dot_product / denominator
    )

    cosine_angle = max(

        -1.0,

        min(1.0, cosine_angle)

    )

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle


# ============================================================
# 9. FUNGSI MENGGAMBAR TITIK LANDMARK
# ============================================================

def draw_point(
    frame,
    landmark,
    label,
    color
):

    height, width = frame.shape[:2]

    if landmark.visibility >= VISIBILITY_THRESHOLD:

        point = landmark_point(
            landmark,
            width,
            height
        )

        cv2.circle(
            frame,
            point,
            7,
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (point[0] + 9, point[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA
        )


# ============================================================
# 10. FUNGSI MENGGAMBAR GARIS KAKI
# ============================================================

def draw_leg_lines(
    frame,
    landmarks,
    hip_id,
    knee_id,
    ankle_id,
    color
):

    height, width = frame.shape[:2]

    hip = landmarks[hip_id]

    knee = landmarks[knee_id]

    ankle = landmarks[ankle_id]

    if (

        hip.visibility >= VISIBILITY_THRESHOLD

        and knee.visibility >= VISIBILITY_THRESHOLD

    ):

        cv2.line(

            frame,

            landmark_point(hip, width, height),

            landmark_point(knee, width, height),

            color,

            3

        )

    if (

        knee.visibility >= VISIBILITY_THRESHOLD

        and ankle.visibility >= VISIBILITY_THRESHOLD

    ):

        cv2.line(

            frame,

            landmark_point(knee, width, height),

            landmark_point(ankle, width, height),

            color,

            3

        )


# ============================================================
# 11. FUNGSI MENAMPILKAN SUDUT LUTUT
# ============================================================

def draw_angle_text(
    frame,
    angle,
    label,
    position,
    color
):

    if angle is None:

        text = f"{label}: NO DATA"

        text_color = (0, 0, 255)

    else:

        text = f"{label}: {angle:.1f} deg"

        text_color = color

    cv2.putText(

        frame,

        text,

        position,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        text_color,

        2,

        cv2.LINE_AA

    )


# ============================================================
# 12. PANEL INFORMASI V8
# ============================================================

def draw_info_panel(
    frame,
    landmarks,
    right_angle,
    left_angle
):

    panel_x = 12

    panel_y = 12

    panel_width = 460

    panel_height = 250

    overlay = frame.copy()

    cv2.rectangle(

        overlay,

        (panel_x, panel_y),

        (
            panel_x + panel_width,
            panel_y + panel_height
        ),

        (0, 0, 0),

        -1

    )

    cv2.addWeighted(

        overlay,

        0.65,

        frame,

        0.35,

        0,

        frame

    )

    cv2.putText(

        frame,

        "V8.1 - AUTOMATIC BASELINE",

        (panel_x + 10, panel_y + 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (0, 255, 255),

        2,

        cv2.LINE_AA

    )

    y = panel_y + 55

    for label, landmark_id in LANDMARK_IDS.items():

        landmark = landmarks[landmark_id]

        information = (

            f"{label}: "

            f"X={landmark.x:.3f} "

            f"Y={landmark.y:.3f} "

            f"VIS={landmark.visibility:.2f}"

        )

        color = (

            (0, 255, 0)

            if landmark.visibility >= VISIBILITY_THRESHOLD

            else (0, 0, 255)

        )

        cv2.putText(

            frame,

            information,

            (panel_x + 10, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.40,

            color,

            1,

            cv2.LINE_AA

        )

        y += 22

    draw_angle_text(

        frame,

        right_angle,

        "RIGHT KNEE",

        (panel_x + 10, panel_y + 205),

        RIGHT_COLOR

    )

    draw_angle_text(

        frame,

        left_angle,

        "LEFT KNEE",

        (panel_x + 245, panel_y + 205),

        LEFT_COLOR

    )


# ============================================================
# 13. FUNGSI STATUS BASELINE
# ============================================================

def draw_baseline_status(
    frame,
    status,
    message,
    baseline_right,
    baseline_left,
    progress
):

    height, width = frame.shape[:2]

    panel_y = 275

    panel_height = 85

    overlay = frame.copy()

    cv2.rectangle(

        overlay,

        (10, panel_y),

        (width - 10, panel_y + panel_height),

        (0, 0, 0),

        -1

    )

    cv2.addWeighted(

        overlay,

        0.70,

        frame,

        0.30,

        0,

        frame

    )

    if status == "CALIBRATED":

        color = (0, 255, 0)

    else:

        color = (0, 255, 255)

    cv2.putText(

        frame,

        f"BASELINE STATUS: {status}",

        (20, panel_y + 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        color,

        2,

        cv2.LINE_AA

    )

    cv2.putText(

        frame,

        message,

        (20, panel_y + 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        TEXT_COLOR,

        1,

        cv2.LINE_AA

    )

    if status != "CALIBRATED":

        baseline_text = (
            f"Calibration progress: "
            f"{progress}/{CALIBRATION_FRAMES} frames"
        )

    else:

        baseline_text = (

            f"RIGHT BASELINE: {baseline_right:.1f} deg"

            f"   |   LEFT BASELINE: {baseline_left:.1f} deg"

        )

    cv2.putText(

        frame,

        baseline_text,

        (20, panel_y + 73),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (255, 255, 255),

        1,

        cv2.LINE_AA

    )


# ============================================================
# 14. HEADER CSV V8.1
# ============================================================

CSV_COLUMNS = [

    "timestamp",

    "frame",

    "pose_detected",

    "RIGHT_KNEE_ANGLE",

    "LEFT_KNEE_ANGLE",

    "RIGHT_BASELINE_ANGLE",

    "LEFT_BASELINE_ANGLE",

    "RIGHT_ANGLE_FROM_BASELINE",

    "LEFT_ANGLE_FROM_BASELINE",

    "CALIBRATION_STATUS"

]

# Seluruh kolom koordinat V8 dipertahankan.

for landmark_name in LANDMARK_IDS:

    CSV_COLUMNS.extend([

        f"{landmark_name}_X",

        f"{landmark_name}_Y",

        f"{landmark_name}_X_PIXEL",

        f"{landmark_name}_Y_PIXEL",

        f"{landmark_name}_VISIBILITY"

    ])


# ============================================================
# 15. MEMBUKA KAMERA DROIDCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    CAMERA_WIDTH

)

cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    CAMERA_HEIGHT

)

if not cap.isOpened():

    pose.close()

    raise RuntimeError(

        f"Kamera index {CAMERA_INDEX} tidak dapat dibuka. "

        "Pastikan DroidCam aktif."

    )


# ============================================================
# 16. MEMBUKA FILE CSV
# ============================================================

frame_number = 0

csv_file = open(

    CSV_PATH,

    mode="w",

    newline="",

    encoding="utf-8"

)

csv_writer = csv.DictWriter(

    csv_file,

    fieldnames=CSV_COLUMNS

)

csv_writer.writeheader()

csv_file.flush()


print("=" * 60)

print("KNEE TRACKING - V8.1 AUTOMATIC BASELINE")

print("=" * 60)

print("Camera Index :", CAMERA_INDEX)

print("Resolution   :", CAMERA_WIDTH, "x", CAMERA_HEIGHT)

print("CSV          :", CSV_PATH.resolve())

print("")

print("FITUR:")

print("- Deteksi enam landmark")

print("- Perhitungan sudut lutut kanan dan kiri")

print("- Visualisasi titik dan garis kaki")

print("- Kalibrasi baseline otomatis")

print("- Pemeriksaan kestabilan 60 frame")

print("- Baseline dikunci setelah berhasil")

print("- Perekaman seluruh data ke CSV")

print("")

print("Tekan Q untuk menghentikan pengujian.")

print("=" * 60)


# ============================================================
# 17. PROSES UTAMA
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success or frame is None:

            print("Frame kamera gagal dibaca.")

            break

        frame_number += 1

        if MIRROR_FRAME:

            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )

        results = pose.process(rgb)

        timestamp = datetime.now().isoformat(

            timespec="milliseconds"

        )

        height, width = frame.shape[:2]

        right_angle = None

        left_angle = None

        landmarks = None

        # ----------------------------------------------------
        # DATA DEFAULT CSV
        # ----------------------------------------------------

        row = {

            "timestamp": timestamp,

            "frame": frame_number,

            "pose_detected": False,

            "RIGHT_KNEE_ANGLE": None,

            "LEFT_KNEE_ANGLE": None,

            "RIGHT_BASELINE_ANGLE": baseline_right,

            "LEFT_BASELINE_ANGLE": baseline_left,

            "RIGHT_ANGLE_FROM_BASELINE": None,

            "LEFT_ANGLE_FROM_BASELINE": None,

            "CALIBRATION_STATUS": calibration_status

        }

        # ----------------------------------------------------
        # 18. DETEKSI POSE
        # ----------------------------------------------------

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            row["pose_detected"] = True

            # ----------------------------------------
            # AMBIL ENAM LANDMARK
            # ----------------------------------------

            right_hip = landmarks[

                mp_pose.PoseLandmark.RIGHT_HIP.value

            ]

            right_knee = landmarks[

                mp_pose.PoseLandmark.RIGHT_KNEE.value

            ]

            right_ankle = landmarks[

                mp_pose.PoseLandmark.RIGHT_ANKLE.value

            ]

            left_hip = landmarks[

                mp_pose.PoseLandmark.LEFT_HIP.value

            ]

            left_knee = landmarks[

                mp_pose.PoseLandmark.LEFT_KNEE.value

            ]

            left_ankle = landmarks[

                mp_pose.PoseLandmark.LEFT_ANKLE.value

            ]

            # ----------------------------------------
            # HITUNG SUDUT LUTUT KANAN
            # ----------------------------------------

            right_angle = calculate_knee_angle(

                right_hip,

                right_knee,

                right_ankle,

                width,

                height

            )

            # ----------------------------------------
            # HITUNG SUDUT LUTUT KIRI
            # ----------------------------------------

            left_angle = calculate_knee_angle(

                left_hip,

                left_knee,

                left_ankle,

                width,

                height

            )

            # ----------------------------------------
            # GAMBAR GARIS KAKI KANAN
            # ----------------------------------------

            draw_leg_lines(

                frame,

                landmarks,

                mp_pose.PoseLandmark.RIGHT_HIP.value,

                mp_pose.PoseLandmark.RIGHT_KNEE.value,

                mp_pose.PoseLandmark.RIGHT_ANKLE.value,

                RIGHT_COLOR

            )

            # ----------------------------------------
            # GAMBAR GARIS KAKI KIRI
            # ----------------------------------------

            draw_leg_lines(

                frame,

                landmarks,

                mp_pose.PoseLandmark.LEFT_HIP.value,

                mp_pose.PoseLandmark.LEFT_KNEE.value,

                mp_pose.PoseLandmark.LEFT_ANKLE.value,

                LEFT_COLOR

            )

            # ----------------------------------------
            # GAMBAR ENAM TITIK DAN SIMPAN KOORDINAT
            # ----------------------------------------

            for label, landmark_id in LANDMARK_IDS.items():

                landmark = landmarks[landmark_id]

                color = (

                    RIGHT_COLOR

                    if "RIGHT" in label

                    else LEFT_COLOR

                )

                draw_point(

                    frame,

                    landmark,

                    label,

                    color

                )

                pixel_x, pixel_y = landmark_point(

                    landmark,

                    width,

                    height

                )

                row[f"{label}_X"] = landmark.x

                row[f"{label}_Y"] = landmark.y

                row[f"{label}_X_PIXEL"] = pixel_x

                row[f"{label}_Y_PIXEL"] = pixel_y

                row[f"{label}_VISIBILITY"] = landmark.visibility

            # ----------------------------------------
            # SIMPAN SUDUT MENTAH
            # ----------------------------------------

            row["RIGHT_KNEE_ANGLE"] = right_angle

            row["LEFT_KNEE_ANGLE"] = left_angle

            # ----------------------------------------
            # TAMPILKAN PANEL INFORMASI V8
            # ----------------------------------------

            draw_info_panel(

                frame,

                landmarks,

                right_angle,

                left_angle

            )

        else:

            cv2.putText(

                frame,

                "POSE NOT DETECTED",

                (25, 45),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 0, 255),

                2,

                cv2.LINE_AA

            )

        # ====================================================
        # 19. KALIBRASI BASELINE OTOMATIS
        # ====================================================

        if calibration_status == "SEARCHING":

            # Kedua sudut harus tersedia bersamaan.

            if (

                right_angle is not None

                and left_angle is not None

            ):

                calibration_buffer.append(

                    (right_angle, left_angle)

                )

            else:

                # Jika salah satu sudut hilang,
                # rangkaian frame stabil dimulai kembali.

                calibration_buffer.clear()

            # ----------------------------------------
            # PERIKSA SETELAH 60 FRAME TERKUMPUL
            # ----------------------------------------

            if len(calibration_buffer) == CALIBRATION_FRAMES:

                right_values = [

                    item[0]

                    for item in calibration_buffer

                ]

                left_values = [

                    item[1]

                    for item in calibration_buffer

                ]

                right_std = statistics.pstdev(

                    right_values

                )

                left_std = statistics.pstdev(

                    left_values

                )

                # ------------------------------------
                # JIKA KEDUA LUTUT STABIL
                # ------------------------------------

                if (

                    right_std <= MAX_CALIBRATION_STD

                    and left_std <= MAX_CALIBRATION_STD

                ):

                    baseline_right = float(

                        statistics.median(right_values)

                    )

                    baseline_left = float(

                        statistics.median(left_values)

                    )

                    calibration_status = "CALIBRATED"

                    calibration_message = (

                        "Baseline berhasil dikunci. "

                        "Perekaman dilanjutkan."

                    )

                    print("")

                    print("=" * 50)

                    print("BASELINE BERHASIL DIKALIBRASI")

                    print(

                        f"RIGHT BASELINE: "

                        f"{baseline_right:.2f} derajat"

                    )

                    print(

                        f"LEFT BASELINE: "

                        f"{baseline_left:.2f} derajat"

                    )

                    print(

                        f"RIGHT SD: {right_std:.2f}"

                    )

                    print(

                        f"LEFT SD: {left_std:.2f}"

                    )

                    print("=" * 50)

                else:

                    # Jika tidak stabil, sistem terus mencari
                    # 60 frame stabil berikutnya.

                    calibration_message = (

                        f"Posisi belum stabil. "

                        f"SD RIGHT={right_std:.1f}, "

                        f"LEFT={left_std:.1f}. "

                        f"Mencari ulang..."

                    )

                    calibration_buffer.clear()

        # ====================================================
        # 20. HITUNG SUDUT RELATIF BASELINE
        # ====================================================

        if calibration_status == "CALIBRATED":

            if (

                right_angle is not None

                and baseline_right is not None

            ):

                row["RIGHT_ANGLE_FROM_BASELINE"] = (

                    right_angle - baseline_right

                )

            if (

                left_angle is not None

                and baseline_left is not None

            ):

                row["LEFT_ANGLE_FROM_BASELINE"] = (

                    left_angle - baseline_left

                )

        # ----------------------------------------------------
        # PERBARUI BASELINE DAN STATUS CSV
        # ----------------------------------------------------

        row["RIGHT_BASELINE_ANGLE"] = baseline_right

        row["LEFT_BASELINE_ANGLE"] = baseline_left

        row["CALIBRATION_STATUS"] = calibration_status

        # ====================================================
        # 21. TAMPILKAN STATUS BASELINE
        # ====================================================

        if calibration_status == "CALIBRATED":

            progress = CALIBRATION_FRAMES

        else:

            progress = len(calibration_buffer)

        draw_baseline_status(

            frame,

            calibration_status,

            calibration_message,

            baseline_right,

            baseline_left,

            progress

        )

        # ====================================================
        # 22. SIMPAN SELURUH DATA KE CSV
        # ====================================================

        csv_writer.writerow(row)

        csv_file.flush()

        # ====================================================
        # 23. TAMPILKAN VIDEO
        # ====================================================

        cv2.imshow(

            "Knee Tracking V8.1 - Automatic Baseline",

            frame

        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# 24. MENUTUP KAMERA DAN FILE
# ============================================================

finally:

    cap.release()

    pose.close()

    csv_file.close()

    cv2.destroyAllWindows()

    print("")

    print("=" * 60)

    print("PENGUJIAN V8.1 SELESAI")

    print("Total frame:", frame_number)

    print("Calibration status:", calibration_status)

    print("Baseline RIGHT:", baseline_right)

    print("Baseline LEFT:", baseline_left)

    print("File CSV:", CSV_PATH.resolve())

    print("=" * 60)