
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import pyrealsense2 as rs
from tensorflow.keras.models import load_model
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

MODEL_PATH = "unet_blood_segmentation_final.h5"

THRESH = 0.1

MIN_AREA_RUIDO   = 400      # área mínima para considerar un blob como posible sangre
AREA_ASPIRAR_ON  = 3000     # área mínima para activar aspiración
AREA_ASPIRAR_OFF = 500      # área máxima para desactivar aspiración (si el blob se reduce o desaparece)
FRAMES_OFF       = 10       # número de frames consecutivos por debajo de AREA_ASPIRAR_OFF para desactivar aspiración (evita parpadeos)

TRACK_DIST_PX    = 150      # distancia máxima en píxeles para asociar blobs entre frames (tracking con C1)
ALPHA_FILTER_3D  = 0.3  # suavizado 3D

VALID_MODES = {"C1", "C2", "C3", "CD", "NO_ROBOT_ALL"}


def filtrar_exp(nuevo, anterior, alpha):
    if anterior is None:
        return np.array(nuevo, dtype=np.float32)
    return alpha * np.array(nuevo, dtype=np.float32) + (1 - alpha) * np.array(anterior, dtype=np.float32)


def clamp_uv(u, v, w, h):
    u = int(np.clip(int(round(u)), 0, w - 1))
    v = int(np.clip(int(round(v)), 0, h - 1))
    return u, v


class BleedingDetectorNode:
    def __init__(self):
        rospy.init_node("bleeding_detector_node")

        # ---- Params base ----
        self.mode = str(rospy.get_param("~mode", "C1")).strip().upper()
        if self.mode not in VALID_MODES:
            rospy.logwarn(f"[bleeding] ~mode='{self.mode}' inválido. Usando C1.")
            self.mode = "C1"

        self.show_age = bool(rospy.get_param("~show_age", False))
        self.show_detection = bool(rospy.get_param("~show_detection", True))

        # ---- Limpieza de máscara (morfología) ----
        self.use_mask_morph = bool(rospy.get_param("~use_mask_morph", True))
        mksz = int(rospy.get_param("~mask_kernel", 5))
        mksz = max(3, mksz if mksz % 2 == 1 else mksz + 1)
        self.MASK_KERNEL = (mksz, mksz)

        # ---- C3: más peso a píxeles nuevos ----
        self.C3_W_MIN = float(rospy.get_param("~c3_w_min", 0.05))
        self.C3_W_MAX = float(rospy.get_param("~c3_w_max", 5.0))
        self.C3_MIN_AGE = int(rospy.get_param("~c3_min_age", 3))

        # ---- Filtro 2D del target publicado ----
        self.ALPHA_FILTER_2D = float(rospy.get_param("~alpha_filter_2d", 0.2))

        # ---- Gating por crecimiento (útil en C3) ----
        self.use_growth_gate = bool(rospy.get_param("~use_growth_gate", True))
        self.MIN_GROWTH_AREA = int(rospy.get_param("~min_growth_area", 80))

        # ---- C2-core params ----
        self.C2_CORE_PERCENTILE = float(rospy.get_param("~c2_core_percentile", 80.0))
        self.C2_CORE_MIN_AREA   = int(rospy.get_param("~c2_core_min_area", 50))

        # ---- Robustez ante fallos breves de detección ----
        self.hold_target_time = float(rospy.get_param("~hold_target_time", 0.5))

        rospy.loginfo(f"[bleeding] mode={self.mode} show_age={self.show_age} show_detection={self.show_detection}")
        rospy.loginfo(f"[bleeding] mask_morph={self.use_mask_morph} mask_kernel={self.MASK_KERNEL}")
        rospy.loginfo(f"[bleeding] alpha2d={self.ALPHA_FILTER_2D} growth_gate={self.use_growth_gate} min_growth_area={self.MIN_GROWTH_AREA}")
        rospy.loginfo(f"[bleeding] C3 clamp: w_min={self.C3_W_MIN} w_max={self.C3_W_MAX}")
        rospy.loginfo(f"[bleeding] C2-core: percentile={self.C2_CORE_PERCENTILE} min_area={self.C2_CORE_MIN_AREA}")
        rospy.loginfo(f"[bleeding] hold_target_time={self.hold_target_time}")

        # ---- Modelo ----
        self.modelo_sangre = load_model(MODEL_PATH, compile=False)

        # ---- ROS pubs ----
        self.pub_centroids_2d = rospy.Publisher("/bleeding/centroids_2d", Float32MultiArray, queue_size=1)
        self.pub_centroides   = rospy.Publisher("/bleeding/sangre_centroides", Point, queue_size=10)
        self.pub_punto_unico  = rospy.Publisher("/bleeding/punto_objetivo", Point, queue_size=1)
        self.pub_imagen       = rospy.Publisher("/bleeding/image_overlay", Image, queue_size=1)
        self.pub_sangre_ok    = rospy.Publisher("/bleeding/sangre_ok", Bool, queue_size=1)

        self.bridge = CvBridge()

        # ---- RealSense ----
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # ---- Estados ----
        self.target_fijo_2d = None
        self.target_fijo_3d = None
        self.target_fijo_2d_cmd = None

        self.last_valid_target_3d = None
        self.last_valid_target_time = rospy.Time(0)

        self.aspirar = False
        self.off_count = 0

        self.blood_age = None  # uint16 HxW
        self.prev_area = None

    # -------------------- Utilidad: republicar último target válido --------------------
    def publish_last_valid_target_if_recent(self):
        if self.last_valid_target_3d is None:
            return False

        dt = (rospy.Time.now() - self.last_valid_target_time).to_sec()
        if dt <= self.hold_target_time:
            self.pub_punto_unico.publish(Point(
                x=float(self.last_valid_target_3d[0]),
                y=float(self.last_valid_target_3d[1]),
                z=float(self.last_valid_target_3d[2])
            ))
            return True

        return False

    # -------------------- Frames --------------------
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    # -------------------- Segmentación --------------------
    def segment(self, color_image):
        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        entrada = cv2.resize(img_rgb, (256, 256))
        entrada = np.expand_dims(entrada, axis=0)

        pred = self.modelo_sangre.predict(entrada, verbose=0)[0]
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        mask = ((pred > THRESH) * 255).astype(np.uint8)
        mask_big = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_big

    # -------------------- Limpieza máscara --------------------
    def clean_mask(self, mask_big):
        if not self.use_mask_morph:
            return mask_big
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.MASK_KERNEL)
        m = cv2.morphologyEx(mask_big, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m

    # -------------------- Age map --------------------
    def update_age(self, mask_big):
        mask_bool = (mask_big > 0)
        if self.blood_age is None or self.blood_age.shape != mask_big.shape:
            self.blood_age = np.zeros(mask_big.shape, dtype=np.uint16)

        self.blood_age[mask_bool] += 1
        self.blood_age[~mask_bool] = 0

    # -------------------- Overlay --------------------
    def apply_overlay(self, color_image, mask_big):
        display_img = color_image.copy()
        red_layer = np.zeros_like(color_image)
        red_layer[:] = [255, 0, 0]  # rojo en BGR

        mask_bool = mask_big > 0
        if np.any(mask_bool):
            roi = display_img[mask_bool]
            blended = cv2.addWeighted(roi, 0.7, red_layer[mask_bool], 0.3, 0)
            display_img[mask_bool] = blended
        return display_img

    # -------------------- C2-core --------------------
    def c2_core_centroid(self, blob_mask_bool):
        if self.blood_age is None:
            return None

        ages = self.blood_age[blob_mask_bool]
        if ages.size == 0:
            return None

        thr = float(np.percentile(ages.astype(np.float32), self.C2_CORE_PERCENTILE))
        core_mask = np.logical_and(blob_mask_bool, self.blood_age.astype(np.float32) >= thr)

        if int(core_mask.sum()) < int(self.C2_CORE_MIN_AREA):
            return None

        ys, xs = np.where(core_mask)
        if xs.size == 0:
            return None

        return (float(xs.mean()), float(ys.mean()))

    # -------------------- C3 centroid ponderado por novedad --------------------
    def c3_new_weighted_centroid(self, blob_mask_bool):
        if self.blood_age is None:
            return None

        # FILTRO: eliminar píxeles demasiado nuevos
        valid_mask = np.logical_and(
            blob_mask_bool,
            self.blood_age >= self.C3_MIN_AGE
        )

        ys, xs = np.where(valid_mask)
        if xs.size == 0:
            return None

        ages = self.blood_age[valid_mask].astype(np.float32)

        # Peso: más peso a lo reciente (pero ya filtrado)
        w = 1.0 / (ages + 1.0)
        w = np.clip(w, self.C3_W_MIN, self.C3_W_MAX)

        s = float(np.sum(w))
        if s <= 1e-9:
            return None

        cx = float(np.sum(xs * w) / s)
        cy = float(np.sum(ys * w) / s)

        return (cx, cy)

    # -------------------- CD (deep point) --------------------
    def deep_point(self, blob_mask_bool):
        if not np.any(blob_mask_bool):
            return None

        mask_u8 = (blob_mask_bool.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        _, _, _, maxLoc = cv2.minMaxLoc(dist)
        return (float(maxLoc[0]), float(maxLoc[1]))

    # -------------------- Blobs --------------------
    def extract_blobs(self, mask_big, depth_frame, display_img):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_big)

        h, w = mask_big.shape[:2]
        puntos_actuales = []
        max_area = 0

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_AREA_RUIDO:
                continue

            max_area = max(max_area, area)

            # C1 geométrico
            cx1, cy1 = float(centroids[i][0]), float(centroids[i][1])
            u1, v1 = clamp_uv(cx1, cy1, w, h)

            # 3D del C1 (solo debug/publicación)
            geom_3d = None
            z = depth_frame.get_distance(u1, v1)
            if z > 0:
                x_3d, y_3d, z_3d = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics, [u1, v1], float(z)
                )
                geom_3d = (x_3d, y_3d, z_3d)

            blob_mask = (labels == i)

            c2 = self.c2_core_centroid(blob_mask)
            c3 = self.c3_new_weighted_centroid(blob_mask)
            cd = self.deep_point(blob_mask)

            # IMPORTANTE:
            # no dibujamos aquí todos los centroides para evitar clutter visual
            # cv2.circle(display_img, (u1, v1), 5, (255, 255, 0), -1)

            puntos_actuales.append({
                "label_id": i,
                "area": area,
                "2d_geom": (cx1, cy1),
                "2d_c2": c2,
                "2d_c3": c3,
                "2d_deep": cd,
                "3d_geom": geom_3d,
            })

        return puntos_actuales, max_area, labels

    # -------------------- Aspiración (condición visual) --------------------
    def update_aspiration_state(self, max_area, total_targets):
        # Si quisieras un modo completamente sin robot:
        # if self.mode == "NO_ROBOT_ALL":
        #     self.aspirar = False
        #     self.off_count = 0
        #     self.pub_sangre_ok.publish(Bool(False))
        #     return

        if not self.aspirar:
            if max_area >= AREA_ASPIRAR_ON and total_targets > 0:
                self.aspirar = True
                self.off_count = 0
        else:
            if (max_area < AREA_ASPIRAR_OFF) or (total_targets == 0):
                self.off_count += 1
                if self.off_count >= FRAMES_OFF:
                    self.aspirar = False
                    self.off_count = 0
            else:
                self.off_count = 0

        self.pub_sangre_ok.publish(Bool(self.aspirar))

    # -------------------- Selección de blob (tracking con C1) --------------------
    def select_blob(self, puntos_actuales):
        if len(puntos_actuales) == 0:
            return None

        largest = max(puntos_actuales, key=lambda p: p["area"])

        if self.target_fijo_2d is None:
            return largest

        last_cx, last_cy = self.target_fijo_2d
        mejor = None
        dist_min = 1e9

        for p in puntos_actuales:
            cx, cy = p["2d_geom"]
            dist = float(np.hypot(cx - last_cx, cy - last_cy))
            if dist < dist_min:
                dist_min = dist
                mejor = p

        if dist_min < TRACK_DIST_PX:
            return mejor

        return largest

    # -------------------- 2D -> 3D --------------------
    def deproject_uv_to_xyz(self, depth_frame, u, v, fallback_xyz=None):
        z = depth_frame.get_distance(int(u), int(v))
        if z > 0:
            x, y, z = rs.rs2_deproject_pixel_to_point(self.intrinsics, [int(u), int(v)], float(z))
            return (x, y, z)
        return fallback_xyz

    # -------------------- Publicación centroides 2D --------------------
    def publish_selected_centroids_2d(self, area, c1, c2, c3, cd):
        """
        Publica Float32MultiArray:
        [t, area, c1x,c1y, c2x,c2y, c3x,c3y, cdx,cdy, aspirar]
        """
        def pack(pt):
            if pt is None:
                return (np.nan, np.nan)
            return (float(pt[0]), float(pt[1]))

        c1x, c1y = pack(c1)
        c2x, c2y = pack(c2)
        c3x, c3y = pack(c3)
        cdx, cdy = pack(cd)

        msg = Float32MultiArray()
        msg.data = [
            float(rospy.Time.now().to_sec()),
            float(area),
            c1x, c1y,
            c2x, c2y,
            c3x, c3y,
            cdx, cdy,
            float(1.0 if self.aspirar else 0.0)
        ]
        self.pub_centroids_2d.publish(msg)

    # -------------------- Publicación + dibujo --------------------
    def publish_and_draw(self, display_img, puntos_actuales, selected_blob, labels, depth_frame):
        # Publicación debug de centroides 3D (si la quieres conservar)
        for p in puntos_actuales:
            if p["3d_geom"] is not None:
                self.pub_centroides.publish(Point(*p["3d_geom"]))

        if selected_blob is None:
            self.target_fijo_2d = None
            self.target_fijo_2d_cmd = None
            self.prev_area = None

            # Mantener el último target válido durante una ventana corta
            self.publish_last_valid_target_if_recent()
            return

        # tracking con C1 geométrico
        self.target_fijo_2d = selected_blob["2d_geom"]

        h, w = display_img.shape[:2]
        lid = selected_blob["label_id"]

        # contorno del blob seleccionado
        sel_mask_bool = (labels == lid)
        sel_u8 = (sel_mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(sel_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_img, contours, -1, (0, 255, 0), 3)

        c1 = selected_blob["2d_geom"]
        c2 = selected_blob["2d_c2"]
        c3 = selected_blob["2d_c3"]
        cd = selected_blob["2d_deep"]

        self.publish_selected_centroids_2d(selected_blob["area"], c1, c2, c3, cd)

        def draw_dot(pt, color, ring=False):
            if pt is None:
                return
            u, v = clamp_uv(pt[0], pt[1], w, h)
            if ring:
                cv2.circle(display_img, (u, v), 10, color, 2)
            else:
                cv2.circle(display_img, (u, v), 5, color, -1)

        # Dibujar solo el target del modo activo
        if self.mode == "C1":
            draw_dot(c1, (255, 255, 0))
        elif self.mode == "C2":
            draw_dot(c2 if c2 is not None else c1, (0, 255, 255))
        elif self.mode == "C3":
            draw_dot(c3 if c3 is not None else c1, (255, 0, 255), ring=True)
        elif self.mode == "CD":
            draw_dot(cd if cd is not None else c1, (255, 255, 255), ring=True)

        # target por modo
        if self.mode == "C1":
            target_2d = c1
        elif self.mode == "C2":
            target_2d = c2 if c2 is not None else c1
        elif self.mode == "C3":
            target_2d = c3 if c3 is not None else c1
        elif self.mode == "CD":
            target_2d = cd if cd is not None else c1
        else:
            target_2d = None

        cv2.putText(display_img, f"mode={self.mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if target_2d is None:
            self.publish_last_valid_target_if_recent()
            return

        # Growth gate (especialmente útil en C3)
        area_now = int(selected_blob["area"])
        if self.use_growth_gate and self.mode == "C3" and self.prev_area is not None:
            growth = area_now - int(self.prev_area)
            if growth < self.MIN_GROWTH_AREA and self.target_fijo_2d_cmd is not None:
                target_2d = (
                    float(self.target_fijo_2d_cmd[0]),
                    float(self.target_fijo_2d_cmd[1])
                )

        self.prev_area = area_now

        # Filtro 2D del target publicado
        if self.ALPHA_FILTER_2D > 0.0:
            self.target_fijo_2d_cmd = filtrar_exp(target_2d, self.target_fijo_2d_cmd, self.ALPHA_FILTER_2D)
            target_cmd = (
                float(self.target_fijo_2d_cmd[0]),
                float(self.target_fijo_2d_cmd[1])
            )
        else:
            target_cmd = target_2d

        u_t, v_t = clamp_uv(target_cmd[0], target_cmd[1], w, h)

        # Cruz verde en target publicado
        cv2.line(display_img, (u_t - 10, v_t), (u_t + 10, v_t), (0, 255, 0), 2)
        cv2.line(display_img, (u_t, v_t - 10), (u_t, v_t + 10), (0, 255, 0), 2)

        xyz = self.deproject_uv_to_xyz(depth_frame, u_t, v_t, fallback_xyz=selected_blob["3d_geom"])
        if xyz is None:
            rospy.logwarn_throttle(1.0, "[bleeding] Depth inválido. Reutilizando último target válido.")
            self.publish_last_valid_target_if_recent()
            return

        coords_suaves = filtrar_exp(xyz, self.target_fijo_3d, ALPHA_FILTER_3D)
        self.target_fijo_3d = coords_suaves

        # Guardar último target válido
        self.last_valid_target_3d = np.array(coords_suaves, dtype=np.float32)
        self.last_valid_target_time = rospy.Time.now()

        self.pub_punto_unico.publish(Point(
            x=float(coords_suaves[0]),
            y=float(coords_suaves[1]),
            z=float(coords_suaves[2])
        ))

        texto = f"XYZ: {coords_suaves[0]:.2f}, {coords_suaves[1]:.2f}, {coords_suaves[2]:.2f}"
        tx = max(10, u_t - 140)
        ty = max(25, v_t - 15)
        cv2.putText(display_img, texto, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(display_img, texto, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def publish_overlay_image(self, display_img):
        self.pub_imagen.publish(self.bridge.cv2_to_imgmsg(display_img, encoding="bgr8"))

    # -------------------- Loop principal --------------------
    def spin(self):
        try:
            while not rospy.is_shutdown():
                color_image, depth_frame = self.get_frames()
                if color_image is None:
                    continue

                mask_big = self.segment(color_image)
                mask_big = self.clean_mask(mask_big)
                self.update_age(mask_big)

                if self.show_age and (self.blood_age is not None):
                    age_vis = cv2.normalize(self.blood_age, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("Age", age_vis)
                else:
                    try:
                        cv2.destroyWindow("Age")
                    except Exception:
                        pass

                display_img = self.apply_overlay(color_image, mask_big)

                puntos_actuales, max_area, labels = self.extract_blobs(mask_big, depth_frame, display_img)
                total = len(puntos_actuales)

                self.update_aspiration_state(max_area, total)

                print(f"\rmode={self.mode} | targets={total} | max_area={max_area} | aspirar={self.aspirar}   ", end="")

                selected_blob = self.select_blob(puntos_actuales)
                self.publish_and_draw(display_img, puntos_actuales, selected_blob, labels, depth_frame)

                self.publish_overlay_image(display_img)

                if self.show_detection:
                    cv2.imshow("Deteccion", display_img)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                else:
                    try:
                        cv2.destroyWindow("Deteccion")
                    except Exception:
                        pass

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    node = BleedingDetectorNode()
    node.spin()
