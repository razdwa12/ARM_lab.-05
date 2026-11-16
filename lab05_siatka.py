import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

KROK = 1.5
MAX_KOSZT = 40
OBROTY = 8
ROZMIAR_MARKERA = 0.07
BOK_KOSTKI = 0.09
MAX_ODL = 3.0

obraz = None
K = None
kostki = {}
bridge = CvBridge()
slownik = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)


# 1. Siatka punktow na mapie kosztow Nav2
def osiagalne(wolne, start):
    odwiedzone = np.zeros_like(wolne)
    odwiedzone[start] = True
    kolejka = deque([start])
    while kolejka:
        w, k = kolejka.popleft()
        for nw, nk in [(w + 1, k), (w - 1, k), (w, k + 1), (w, k - 1)]:
            if 0 <= nw < wolne.shape[0] and 0 <= nk < wolne.shape[1]:
                if wolne[nw, nk] and not odwiedzone[nw, nk]:
                    odwiedzone[nw, nk] = True
                    kolejka.append((nw, nk))
    return odwiedzone


def wyznacz_punkty(nav):
    mapa = nav.getGlobalCostmap()
    m = mapa.metadata
    siatka = np.array(mapa.data).reshape(m.size_y, m.size_x)
    krok = int(KROK / m.resolution)

    start_w = int(-m.origin.position.y / m.resolution)
    start_k = int(-m.origin.position.x / m.resolution)
    dostepne = osiagalne(siatka < MAX_KOSZT, (start_w, start_k))

    punkty = []
    for w in range(0, m.size_y, krok):
        for k in range(0, m.size_x, krok):
            if dostepne[w, k]:
                x = m.origin.position.x + k * m.resolution
                y = m.origin.position.y + w * m.resolution
                punkty.append((x, y))
    return punkty


# 2. Kolejnosc: zawsze najblizszy punkt
def uloz_kolejnosc(punkty):
    kolejnosc = []
    obecny = (0.0, 0.0)
    while punkty:
        obecny = min(punkty, key=lambda p: math.dist(p, obecny))
        punkty.remove(obecny)
        kolejnosc.append(obecny)
    return kolejnosc


# 3. Markery do RViz
def marker(nav, typ, ns, id, x, y, z, rozmiar, r, g, b):
    m = Marker()
    m.header.frame_id = 'map'
    m.header.stamp = nav.get_clock().now().to_msg()
    m.type = typ
    m.ns = ns
    m.id = id
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = rozmiar
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
    return m


def pokaz_siatke(nav, pub, punkty):
    m = marker(nav, Marker.SPHERE_LIST, 'siatka', 0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.8, 0.0)
    m.points = [Point(x=x, y=y) for x, y in punkty]
    pub.publish(m)


def pokaz_kostki(nav, pub):
    for id, (x, y, n) in policz_kostki().items():
        pub.publish(marker(nav, Marker.CUBE, 'kostki', id, x, y, 0.1, 0.2, 0.0, 0.0, 1.0))
        napis = marker(nav, Marker.TEXT_VIEW_FACING, 'id', id, x, y, 0.5, 0.4, 1.0, 1.0, 1.0)
        napis.text = f'ID {id}'
        pub.publish(napis)


# 4. Wykrywanie markerow ArUco
def odbierz_obraz(msg):
    global obraz
    obraz = msg


def odbierz_info(msg):
    global K
    K = np.array(msg.k).reshape(3, 3)


def znajdz_markery(szary):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detektor = cv2.aruco.ArucoDetector(slownik, cv2.aruco.DetectorParameters())
        rogi, ids, _ = detektor.detectMarkers(szary)
    else:
        rogi, ids, _ = cv2.aruco.detectMarkers(szary, slownik)
    return rogi, ids


def wykryj(tf_buffer):
    if obraz is None or K is None:
        return
    szary = cv2.cvtColor(bridge.imgmsg_to_cv2(obraz, 'bgr8'), cv2.COLOR_BGR2GRAY)
    rogi, ids = znajdz_markery(szary)
    if ids is None:
        return

    tf = tf_buffer.lookup_transform('map', 'camera_rgb_optical_frame', Time())
    s = ROZMIAR_MARKERA / 2
    narozniki = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    for rog, id in zip(rogi, ids.flatten()):
        _, rvec, tvec = cv2.solvePnP(narozniki, rog.reshape(4, 2), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        R, _ = cv2.Rodrigues(rvec)
        srodek = tvec.flatten() - R[:, 2] * BOK_KOSTKI / 2
        odl = float(np.linalg.norm(tvec))
        if odl > MAX_ODL:
            continue

        p = PointStamped()
        p.point.x, p.point.y, p.point.z = map(float, srodek)
        p = do_transform_point(p, tf).point

        kostki.setdefault(int(id), []).append((p.x, p.y, odl))
        print(f'  ID {id}: ({p.x:.2f}, {p.y:.2f}), odl. {odl:.2f} m')


# 5. Srednia wazona pozycji kazdej kostki
def policz_kostki():
    wynik = {}
    for id, pomiary in kostki.items():
        p = np.array(pomiary)
        wagi = 1 / p[:, 2] ** 2
        wynik[id] = (np.average(p[:, 0], weights=wagi), np.average(p[:, 1], weights=wagi), len(p))
    return wynik


# 6. Jazda i rozgladanie sie
def czekaj(nav, sekundy):
    koniec = time.time() + sekundy
    while time.time() < koniec:
        rclpy.spin_once(nav, timeout_sec=0.05)


def jedz(nav, x, y):
    cel = PoseStamped()
    cel.header.frame_id = 'map'
    cel.pose.position.x = x
    cel.pose.position.y = y
    cel.pose.orientation.w = 1.0
    nav.goToPose(cel)
    while not nav.isTaskComplete():
        pass
    return nav.getResult() == TaskResult.SUCCEEDED


def rozejrzyj_sie(nav, tf_buffer, pub):
    for i in range(OBROTY):
        czekaj(nav, 1.0)
        wykryj(tf_buffer)
        pokaz_kostki(nav, pub)
        nav.spin(spin_dist=2 * math.pi / OBROTY)
        while not nav.isTaskComplete():
            pass


# 7. Program glowny
def main():
    rclpy.init()
    nav = BasicNavigator()
    nav.create_subscription(Image, '/camera/image_raw', odbierz_obraz, 1)
    nav.create_subscription(CameraInfo, '/camera/camera_info', odbierz_info, 1)
    pub_siatka = nav.create_publisher(Marker, '/punkty_siatki', 10)
    pub_kostki = nav.create_publisher(Marker, '/kostki', 10)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, nav)
    nav.waitUntilNav2Active(localizer='bt_navigator')

    punkty = uloz_kolejnosc(wyznacz_punkty(nav))
    print(f'Punktow siatki: {len(punkty)}')

    for i, (x, y) in enumerate(punkty):
        pokaz_siatke(nav, pub_siatka, punkty)
        print(f'Punkt {i + 1}/{len(punkty)}: ({x:.2f}, {y:.2f})')
        if jedz(nav, x, y):
            rozejrzyj_sie(nav, tf_buffer, pub_kostki)
        if len(kostki) >= 5:
            break

    print('Wynik:')
    for id, (x, y, n) in sorted(policz_kostki().items()):
        print(f'  ID {id}: x = {x:.2f}, y = {y:.2f} ({n} detekcji)')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
