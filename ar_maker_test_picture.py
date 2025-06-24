import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import datetime
import time
import os

def try_camera_calibration(chessboard_size=(9, 6), square_size=25.0):
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

# 初期設定
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

cameraMatrix, distCoeffs = try_camera_calibration()
if cameraMatrix is None:
    cameraMatrix = np.array([[639.87721705,   0.        , 330.12073612],
                             [  0.        , 643.69687408, 208.61588364],
                             [  0.        ,   0.        ,   1.        ]], dtype=np.float64)
    distCoeffs = np.zeros(5)

marker_length = 31.0  # mm

print("📂 input_images フォルダを監視中...（Ctrl+Cで終了）")

while True:
    image_files = sorted(glob.glob("input_images/*.jpg"), key=os.path.getmtime)
    if not image_files:
        print("画像が見つかりません。")
        time.sleep(1)
        continue

    latest_image = image_files[-1]
    frame = cv2.imread(latest_image)
    if frame is None:
        time.sleep(1)
        continue

    frame_undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)
    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
    coords = {7: (0.0, 0.0), 8: (0.0, 0.0), 27: (0.0, 0.0)}  # デフォルトを (0,0) に

    if ids is not None:
        retval, rvecs, tvecs = aruco.estimatePoseSingleMarkers(
            corners, marker_length, cameraMatrix, distCoeffs)

        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id not in coords:
                continue

            rvec = rvecs[i][0]
            tvec = tvecs[i][0].reshape(3, 1)

            R, _ = cv2.Rodrigues(rvec)
            corners_ = corners[i][0]
            d1 = np.linalg.norm(corners_[0] - corners_[1])
            d2 = np.linalg.norm(corners_[1] - corners_[2])
            d3 = np.linalg.norm(corners_[2] - corners_[3])
            d4 = np.linalg.norm(corners_[3] - corners_[0])
            avg_pixel_length = (d1 + d2 + d3 + d4) / 4
            fx = cameraMatrix[0, 0]
            estimated_distance_mm = (fx * marker_length) / avg_pixel_length

            center_2d = corners_[0:4].mean(axis=0)
            cx, cy = int(center_2d[0]), int(center_2d[1])
            K_inv = np.linalg.inv(cameraMatrix)
            pixel_vec = np.array([cx, cy, 1], dtype=np.float64).reshape(3, 1)
            norm_camera_vec = K_inv @ pixel_vec
            camera_coord_3d = norm_camera_vec * estimated_distance_mm
            x3d, y3d, z3d = camera_coord_3d.flatten()
            coords[marker_id] = (x3d, y3d)

            aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, marker_length / 2)

    # 出力
    x1, y1 = coords[7]
    x2, y2 = coords[8]
    x3, y3 = coords[27]
    now = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{now}] (x1,y1)=({x1:.1f},{y1:.1f}) (x2,y2)=({x2:.1f},{y2:.1f}) (x3,y3)=({x3:.1f},{y3:.1f})")

    # 表示（任意）
    cv2.imshow("Aruco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1)  # 同じ画像でも1秒おきに処理

cv2.destroyAllWindows()
