import re
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

MARKER_SIZE = 0.07
CUBE_SIZE = 0.09
HFOV = 1.02974
CAMERA_FRAME = 'camera_rgb_optical_frame'
GT_FILE = '/root/.ros/arm05_sim/logs/latest.txt'


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_ground_truth(path):
    gt = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'Aruco (\d+): x: ([-\d.e]+) y: ([-\d.e]+)', line)
            if m:
                gt[int(m.group(1))] = np.array([float(m.group(2)), float(m.group(3))])
    return gt


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.K = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.gt = load_ground_truth(GT_FILE)
        self.get_logger().info(f'ground truth: {self.gt}')

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
        else:
            self.detector = None
            self.params = cv2.aruco.DetectorParameters_create()

        s = MARKER_SIZE / 2
        self.obj_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

    def lookup_camera(self):
        for frame in ['map', 'odom']:
            try:
                t = self.tf_buffer.lookup_transform(frame, CAMERA_FRAME, Time())
                tr = t.transform.translation
                return frame, quat_to_rot(t.transform.rotation), np.array([tr.x, tr.y, tr.z])
            except Exception:
                pass
        return None, None, None

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]

        # Zadanie 1 - macierz kamery z FOV
        if self.K is None:
            f = (w / 2) / np.tan(HFOV / 2)
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
            self.get_logger().info(f'obraz {w}x{h}, fx = {f:.1f}, frame_id = {msg.header.frame_id}')

        # Zadanie 2 - detekcja
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            frame, R_wc, t_wc = self.lookup_camera()

            for c, marker_id in zip(corners, ids.flatten()):
                # Zadanie 3 - poza tagu w ramce kamery
                pts = c.reshape(4, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(self.obj_points, pts, self.K, None,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                R_ct, _ = cv2.Rodrigues(rvec)
                center_cam = tvec.flatten() - CUBE_SIZE / 2 * R_ct[:, 2]

                # Zadanie 4 - pozycja kostki w mapie i porownanie z latest.txt
                if frame is None:
                    self.get_logger().warn('brak TF map/odom -> kamera', throttle_duration_sec=2.0)
                    continue
                p = R_wc @ center_cam + t_wc
                text = f'ID {marker_id} [{frame}]: x={p[0]:.2f} y={p[1]:.2f}, d={np.linalg.norm(tvec):.2f} m'
                if marker_id in self.gt:
                    gt = self.gt[marker_id]
                    err = np.linalg.norm(p[:2] - gt)
                    text += f' | GT x={gt[0]:.2f} y={gt[1]:.2f} | blad={err:.2f} m'
                self.get_logger().info(text, throttle_duration_sec=1.0)

        small = cv2.resize(img, (960, 540))
        cv2.imshow('aruco', small)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()