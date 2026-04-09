
# Algoritmo de Detección y Caracterización del Sangrado

Este repositorio contiene el código desarrollado para la detección y segmentación de sangre en imágenes laparoscópicas usando un modelo de segmentación tipo U-Net. Se han desarrollado dos variantes del código: 
-	Segmentación fuera de línea: las imágenes que se procesan provienen de la lectura de un vídeo de imágenes de cirugía laparoscópica (`detección_sangre_video_node.py`).
-	Segmentación en tiempo real: las imágenes se capturan en tiempo real de una cámara, en particular de una cámara de profundidad Intel Real Sense D405 (`deteccion_sangre_video_node.py`).

El funcionamiento del código en ambos casos es el siguiente: 
1.	Carga del modelo (.h5). Este modelo se ha generado entrenando la red con la base de datos proporcionada en [1], que contiene 750 imágenes de cirugía ginecológica anotadas con máscaras de sangrado.
2.	Carga de las imágenes, bien de un vídeo en el caso de segmentación fuera de línea, o de la cámara endoscópica en el caso de la segmentación en tiempo real. 
3.	Procesamiento de la imagen:  incluye las fases de (1) preprocesamiento, (2) inferencia del mapa de probabilidad de sangrado, (3) umbralización para convertir el mapa de probabilidad en una máscara binaria y (4) reescalado para transformar la máscara al tamaño original de la imagen. 
4.	Filtrado por área: se eliminan las áreas menores de un determinado umbral. 
5.	Persistencia temporal: se aplica un umbral de persistencia de sangrado para evitar ruido en la detección. 
6.	Características geométricas: cálculo del centroide de cada región de sangre detectada. 
7.	Publicación en ROS: publicación de la máscara de sangrado y la información geométrica de las regiones detectadas (área y centroide). 


### Requisitos del Sistema:

- **Sistema Operativo**: Ubuntu 22.04
- **Lenguaje de Programación**: Python 3

### Librerías Utilizadas:

1. `tensorflow`: Para el desarrollo de modelos de detección y segmentación.
2. `opencv`: Para el procesamiento de imágenes y video.
3. `matplotlib`: Para la visualización de resultados.

### Instalación de Librerías:
```bash
    pip install tensorflow
    pip install matplotlib
    apt install python3-opencv
    pip install scikit-learn
    pip install seaborn

```

### Resultados
<a href="https://www.youtube.com/watch?v=_BtoFf8n64o" target="_blank">
  <img src="https://img.youtube.com/vi/_BtoFf8n64o/maxresdefault.jpg" alt="Blood segmentation in real surgical images" width="400">
</a>

### Referencias
[1] Rabbani, N., Seve, C., Bourdel, N. & Bartoli, A.. (2022). Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation. Proceedings of The 5th International Conference on Medical Imaging with Deep Learning, in Proceedings of Machine Learning Research 172:961-974. 
