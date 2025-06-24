import cv2
import cv2.aruco as aruco
import numpy as np
import datetime
from collections import defaultdict

# --- キャリブレーション用関数 ---
def try_camera_calibration(chessboard_size=(9, 6), square_size=25.0):
    import glob
    images = glob.glob("calib_images/*.jpg")
    if len(images) == 0:
        print("⚠️ calib_images/ が空。仮のパラメータを使用します。")
        return None, None

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints, imgpoints = [], []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) < 3:
        print("⚠️ キャリブ画像が少なすぎます。仮のパラメータを使用します。")
        return None, None

    ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("✅ カメラキャリブレーション成功")
    return cameraMatrix, distCoeffs

# --- 射影変換 ---
pts1 = np.array([(171, 275), (434, 272), (63, 397), (488, 405)], dtype=np.float32)
pts2 = np.array([(-434, 1520), (175, 1520), (-434, 912), (175, 912)], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts1, pts2)

# --- Kalman フィルタークラス ---
class SimpleKalman:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return float(prediction[0]), float(prediction[1])

# --- ArUco検出器設定 ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# --- カメラパラメータ ---
cameraMatrix, distCoeffs = try_camera_calibration()
if cameraMatrix is None:
    cameraMatrix = np.array([[639.87721705, 0., 330.12073612],
                             [0., 643.69687408, 208.61588364],
                             [0., 0., 1.]], dtype=np.float64)
    distCoeffs = np.zeros(5)

# --- KalmanフィルタをIDごとに保持 ---
kalman_filters = defaultdict(SimpleKalman)

# --- マーカーサイズ（mm） ---
marker_length = 31.0

def detect_aruco_filtered_real_positions(image_bytes: bytes) -> list:
    """
    画像バイナリから ArUco マーカー ID 7,8,27 の実座標を透視変換し、
    Kalman フィルターで平滑化して (x, y) のリストで ID 昇順（7,8,27）に返す。
    マーカーが検出されなければ (0.0, 0.0) を返す。
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("画像の読み込みに失敗しました")

    frame_undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)
    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
<<<<<<< HEAD
    coords = {7: (0.0, 0.0), 8: (0.0, 0.0), 27: (0.0, 0.0)}
=======
    target_ids = [7, 8, 27]
    results = {tid: (0.0, 0.0) for tid in target_ids}
>>>>>>> Anthony

    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id in results:
                center_2d = corners[i][0].mean(axis=0).astype(np.float32).reshape(-1, 1, 2)
                real = cv2.perspectiveTransform(center_2d, M)
                x, y = real[0][0]
                x_filt, y_filt = kalman_filters[marker_id].update(x, y)
                results[marker_id] = (x_filt, y_filt)

<<<<<<< HEAD
            center_2d = corners[i][0].mean(axis=0).astype(np.float32).reshape(-1, 1, 2)
            real = cv2.perspectiveTransform(center_2d, M)
            x, y = real[0][0]
            # Kalmanで平滑化
            x_filt, y_filt = kalman_filters[marker_id].update(x, y)
            coords[marker_id] = (x_filt, y_filt)

    return [coords[marker_id[0][0]], coords[marker_id[1][0]], coords[marker_id[2][0]]]
=======
    # ID順（7, 8, 27）でリスト化
    return [results[tid] for tid in target_ids]
>>>>>>> Anthony
