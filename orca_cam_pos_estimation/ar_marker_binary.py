import cv2
import cv2.aruco as aruco
import numpy as np
from collections import defaultdict

# ===== 射影変換行列 =====
pts1 = np.array([(121, 333), (185, 96), (409, 102), (481, 334)], dtype=np.float32)
pts2 = np.array([(-304, 913), (-304, 1842), (304, 1842), (304, 913)], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts1, pts2)

# ===== カメラパラメータ =====
cameraMatrix = np.array([[1452.1857, 0., 626.778949],
                         [0., 1452.27677, 451.452723],
                         [0., 0., 1.]], dtype=np.float64)
distCoeffs = np.array([0.09215690, 0.51481344, 0.02651601, -0.00706794, -3.74512909], dtype=np.float64)

# ===== ArUco設定 =====
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ===== Kalman フィルタークラス定義 =====
class SimpleKalman:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
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

# ===== Kalmanフィルター保持辞書 =====
kalman_filters = defaultdict(SimpleKalman)

# ===== メイン関数 =====
def detect_aruco_and_get_real_positions(image_path: str,target_ids=[7, 8, 29]) -> dict:
    """
    ArUcoマーカーの補正済み位置を射影変換し、Kalmanフィルターで平滑化して返す。
    出力: {id: (X, Y)} ← 実世界座標（mm）。未検出は (0.0, 0.0)
    スケーリングは x, y 共に 73/84 を適用。
    """
    #nparr = np.frombuffer(image_bytes, np.uint8)
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("画像読み込みに失敗しました")

    # 歪み補正とグレースケール変換
    undistorted = cv2.undistort(img, cameraMatrix, distCoeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    # ArUco 検出
    corners, ids, _ = detector.detectMarkers(gray)
    results = {tid: (0.0, 0.0) for tid in target_ids}

    if ids is not None:
        aruco.drawDetectedMarkers(undistorted, corners, ids)
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id in results:
                # マーカー中心座標の計算と透視変換
                center_2d = corners[i][0].mean(axis=0).astype(np.float32).reshape(-1, 1, 2)
                real = cv2.perspectiveTransform(center_2d, M)
                x, y = real[0][0]

                # Kalmanフィルターで平滑化
                x_filt, y_filt = kalman_filters[marker_id].update(x, y)

                # スケーリングして登録（変更しないとの指定）
                results[marker_id] = ((73 * y_filt / 84)/1000, (-1)*(73 * x_filt / 84)/1000)

    return results
