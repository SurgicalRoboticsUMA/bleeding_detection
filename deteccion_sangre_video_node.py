#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Float32MultiArray


VIDEO_PATH = "/root/catkin_ws/src/bleeding/src/nefrectomia2.mp4"
MODEL_PATH = "/root/catkin_ws/src/bleeding/src/unet_blood_segmentation_final.h5"
THRESH = 0.8    # umbral a partir del cual se considera sangre

import tensorflow as tf
print("Configurando GPU...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

modelo_sangre = load_model(MODEL_PATH, compile=False)

rospy.init_node("bleending_detector_video_node")
bridge = CvBridge()

pub_overlay = rospy.Publisher("/bleeding/video_overlay", Image, queue_size=1)
pub_mask = rospy.Publisher("/bleeding/blood_mask", Image, queue_size=1)  #Máscara binaria: 0=no sangre, 255=sangre
pub_blood = rospy.Publisher("/bleeding/blood_info", Float32MultiArray, queue_size=1)
pub_age = rospy.Publisher("/bleeding/blood_age", Image, queue_size=1)

blood_age = None  # variable global para almacenar la edad de la sangre detectada

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

rate = rospy.Rate(30)   # 30 FPS

try:

    print("Video abierto:", cap.isOpened())
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reinicia al final
            continue
            print("Error 1")

        # --- segmentación ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        entrada = cv2.resize(img_rgb, (256, 256))
        entrada = np.expand_dims(entrada, axis=0)

        pred = modelo_sangre.predict(entrada, verbose=0)[0]
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        
        # Conocer los valores mínimo y máximo de la predicción
        #print("pred min/max:", pred.min(), pred.max())


        mask = (pred > THRESH).astype(np.uint8) * 255
        mask_big = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ===== FILTRADO POR ÁREA (elimina venillas) =====
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_big)
        #num_labels: número de manchas de sangre encontradas (incluyendo el fondo)
        #labels: imagen donde cada píxel tiene el número de la mancha a la que pertenece (0 = fondo, 1 = primera mancha, etc.)
        #stats: matriz con información de cada mancha (x, y, ancho, alto, área)

        # Crear una máscara limpia donde solo se mantengan las manchas con área suficiente
        clean = np.zeros_like(mask_big)
        MIN_AREA = 4000  # ajusta: 2000–6000 para 1080p

        for i in range(1, num_labels):  # saltar fondo
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA:
                clean[labels == i] = 255

        mask_big = clean
        num_labels_clean, labels_clean, stats_clean, centroids_clean = cv2.connectedComponentsWithStats(clean)  

        # Array con la info de cada mancha: ID, área, centroide (x,y)
        blood = []

        for i in range(1, num_labels_clean):  # saltar fondo
            area = int(stats_clean[i, cv2.CC_STAT_AREA])
            cx = float(centroids_clean[i][0])
            cy = float(centroids_clean[i][1])

            blood.append({
                "id": i,
                "area": area,
                "centroid": (cx, cy)
            })
        
        #print("Manchas detectadas:", blood)
        
        #------------------------------------------------------------

        # --- overlay rojo ---
        overlay = frame.copy()
        overlay[mask_big == 255] = [255, 0, 0]  # rojo (BGR)
        superpuesta = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Dibujar centroides en la imagen
        for i in range(1, num_labels_clean):  # saltar fondo
            cx, cy = centroids_clean[i]
            # Dibujar un círculo en el centroide
            cv2.circle(superpuesta, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            # Opcional: escribir el ID de la mancha
            cv2.putText(superpuesta, f"{i}", (int(cx)+8, int(cy)+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        #------------------------------------------------------------
        # --- publicar en ROS ---
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = "video_frame"

        # Publicar imagen superpuesta
        msg_overlay = bridge.cv2_to_imgmsg(superpuesta, encoding="bgr8")
        msg_overlay.header = h
        pub_overlay.publish(msg_overlay)

        # Publicar máscara binaria
        msg_mask = bridge.cv2_to_imgmsg(mask_big, encoding="mono8")
        msg_mask.header = h
        pub_mask.publish(msg_mask)
        
        # Publicar info de la detección de sangre (como un array de Float32MultiArray)
        blood = []   # array plano

        # Cálculo del centroide en base a la edad del pixel
        # Cálculo de la edad de cada pixel
        if blood_age is None or blood_age.shape != clean.shape:
            # Se inicializa una sola vez con el tamaño de la máscara
            blood_age = np.zeros_like(clean, dtype=np.uint16)

        # Sumar 1 donde hay mancha grande
        blood_age[clean == 255] += 1

        # Resetear donde NO hay mancha grande
        blood_age[clean == 0] = 0
        
        # OPCIONES PARA EL CÁLCULO DEL CENTROIDE PONDERADO:
        # A) Dar más peso a píxeles “nuevos” (más recientes)
        weights = 1.0 / (blood_age.astype(np.float32) + 1.0)

        # B) Dar más peso a píxeles “viejos” (persistentes)
        # weights = (blood_age.astype(np.float32) + 1.0)

        # C) C) Control fino con curva: Usa un exponente para ajustar agresividad:
        # Favorecer recientes (decay): w = (age+1)^(-p)
        # Favorecer antiguos (growth): w = (age+1)^(+p)
        # p = 0.5  # exponente de control de agresividad
        # weights = np.power(blood_age.astype(np.float32) + 1.0, -p)

        for i in range(1, num_labels_clean):
            area = int(stats_clean[i, cv2.CC_STAT_AREA])
            #Centroide geométrico
            cx = float(centroids_clean[i][0])
            cy = float(centroids_clean[i][1])

            mask_i = (labels_clean == i)
            ys, xs = np.where(mask_i) # Coordenadas de los píxeles de la mancha
            w = weights[mask_i] # Pesos de esos píxeles
            #Centroide ponderado por edad
            cx_w = np.sum(xs * w) / np.sum(w) 
            cy_w = np.sum(ys * w) / np.sum(w)
            # Añadir los valores en orden, sin diccionarios
            blood.extend([i, area, cx, cy, cx_w, cy_w])


        msg_blood = Float32MultiArray()
        msg_blood.data = blood
        pub_blood.publish(msg_blood)

        

        #print("Edad de sangre (min/max):", blood_age.min(), blood_age.max())
        blood_age_vis = cv2.normalize(blood_age, None, 0, 255, cv2.NORM_MINMAX)
        blood_age_vis = blood_age_vis.astype(np.uint8)

        msg_age = bridge.cv2_to_imgmsg(blood_age_vis, encoding="mono8")
        #msg_age.header = h
        pub_age.publish(msg_age)


        # (opcional) mostrar en pantalla
        print("Procesando frame…")
        cv2.imshow("Overlay sangre (video)", superpuesta)
        if cv2.waitKey(1) == 27:
           break
        
        rate.sleep()

finally:
    cap.release()
    cv2.destroyAllWindows()
