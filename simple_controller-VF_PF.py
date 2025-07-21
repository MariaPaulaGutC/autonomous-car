#Navegación Autonoma\\MR4010.10
## Proyecto Final
##Profesor titular: Dr. David Antonio Torres
##Profesor evaluador:Julio César Salgado Ramírez
###Luis Eduardo Calvillo Corona A00464759
###Juan Manuel Hernández Carrillo A01785878
###Marco Antonio Arellano Hernández A00377571
###María Paula Gutiérrez Cervantes A01747706

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import math

#Ruta del model 2 para la detección de objetos usando yolo
model2_path=r'C:\Users\leduc\Documents\ITESM\MNA\Navegacion\MR4010ProyectoFinal2025\yolov10m.pt'
model2 = YOLO(model2_path)  #yolo11n, yolo12s

#Para tener las imágenes en rgb
def rgba_to_rgb(image_rgba):
    if image_rgba.shape[2] == 4:
       image_rgb = image_rgba[:, :, :3]
    else:
        image_rgb = image_rgba
    return image_rgb # Retorna una imagen con 3 canales RGB

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    #return image

    image_rgb = rgba_to_rgb(image) 
    return image_rgb # Retorna la imagen en formato RGB (3 canales)

def get_image_disp(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing para Hough
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def detect_edges(gray_img, low_threshold=50, high_threshold=100):
    return cv2.Canny(gray_img, low_threshold, high_threshold,apertureSize=3)

# Hough line detection (Probabilistic)

def detect_lines(edges, rho=1, theta=np.pi/180, threshold=30,
                 min_line_length=20, max_line_gap=50):
    return cv2.HoughLinesP(
        edges, rho, theta, threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

def region_of_interest(img):

    mask = np.zeros_like(img)
    mask_color = 255 if len(img.shape) == 2 else (255,) * img.shape[2]

    vertices = np.array([[
        (60, 160),
        (60, 80),
        (280, 80),
        (280, 160)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, mask_color)

    return cv2.bitwise_and(img, mask)

def region_of_interestH(img):

    mask = np.zeros_like(img)
    mask_color = 255 if len(img.shape) == 2 else (255,) * img.shape[2]

    vertices = np.array([[
    (150, 100),  
    (150, 160),  
    (320, 160), 
    (320, 100)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, mask_color)

    return cv2.bitwise_and(img, mask)

# Draw Hough lines on image
def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    line_img = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def draw_info_overlay(image, lidar_data, angulo_prom, speed):
   # Dibuja un recuadro negro con información del LiDAR, velocidad y dirección.
    overlay = image.copy()
    h, w, _ = image.shape

    # Define la región del recuadro (puedes ajustar la posición y tamaño)
    box_top_left = (10, 10)
    box_bottom_right = (260, 90)
    cv2.rectangle(overlay, box_top_left, box_bottom_right, (0, 0, 0), -1)  # fondo negro

    # Define el texto a mostrar
    text_lines = [
        f"Steering: {angulo_prom:.1f} rad",
        f"Speed {speed:.3f} km/h",
        f"LiDAR: {lidar_data}"  # Debes generar un resumen legible
    ]

    # Dibujar cada línea de texto
    y0 = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * 20
        cv2.putText(overlay, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay

frames_without_detection = 0
decay_rate = 0.01 # Ajusta este valor para la velocidad de decaimiento

def set_steering_angle_from_hough_lines(lines):
    global angle, steering_angle, frames_without_detection, decay_rate

    if lines is None or not lines.any():
        print("No se detectaron líneas Hough.")
        frames_without_detection += 1
        steering_angle *= (1 - decay_rate) # Reducir gradualmente el ángulo
        angle = steering_angle
        print(f"Perdiendo carriles, ángulo actual: {steering_angle:.4f}")
        return

    frames_without_detection = 0

    line_angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            line_angles.append(angle_deg)

    if not line_angles:
        print("No se encontraron ángulos de línea válidos.")
        return

    average_angle_degrees = np.mean(line_angles)
    steering_adjustment = -average_angle_degrees * 0.05

    desired_steering_angle = steering_angle + steering_adjustment

    # --- Aplicar las limitaciones de dirección ---
    max_steering_change = 0.1
    if (desired_steering_angle - steering_angle) > max_steering_change:
        desired_steering_angle = steering_angle + max_steering_change
    if (desired_steering_angle - steering_angle) < -max_steering_change:
        desired_steering_angle = steering_angle - max_steering_change

    max_steering_angle = 0.1
    if desired_steering_angle > max_steering_angle:
        desired_steering_angle = max_steering_angle
    elif desired_steering_angle < -max_steering_angle:
        desired_steering_angle = -max_steering_angle

    steering_angle_hough = desired_steering_angle
    
    print(f"Ángulo de dirección aplicado (rad): {steering_angle_hough:.4f}")
    return(steering_angle_hough)

#Display image 
def display_image(display, image):
    disp_w = display.getWidth()
    disp_h = display.getHeight()
#Redimensiona la imagen al tamaño exacto del Display
    image_resized = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    image_ref = display.imageNew(
        image_resized.tobytes(),
        Display.RGB,
        width=disp_w,
        height=disp_h,
    )
    display.imagePaste(image_ref, 0, 0, False)

def display_image_test(display, image):
    image_ref = display.imageNew(
        image.tobytes(),
        Display.RGB,
        width=image.shape[1],
        height=image.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 10 # en km/h
v_max=speed/3.6 # en m/s

# set target speed
def set_speed(distancia, d_max=40, d_min=0.5, v_max=v_max):
    global speed           
    if distancia >= d_max:
        speed_drive=3.6*v_max
        return speed_drive
    elif distancia <= d_min:
        speed_drive=0.0
        return speed
    else:
        speed_drive=3.6*v_max * (distancia - d_min) / (d_max - d_min)
        return speed_drive

#La función para fijar el ángulo de conducción
def set_steering_angle(wheel_angle):
    global angle, steering_angle

    print(f"Predicted: {wheel_angle:.3f} | Previous: {steering_angle:.3f}")

    # Limitar cambio por paso (±0.1 rad)
    max_delta = 0.1
    delta = wheel_angle - steering_angle

    if delta > max_delta:
        angle = steering_angle + max_delta
    elif delta < -max_delta:
        angle = steering_angle - max_delta
    else:
        angle = wheel_angle

    # Limitar el rango total de dirección (±0.5 rad)
    angle = max(-0.5, min(0.5, angle))

    # Actualiza el valor global para el siguiente paso
    steering_angle = angle

    print(f"→ Applied angle: {angle:.3f} rad")   

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

def change_steer_angle_new(pred_angle):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + pred_angle
    manual_steering = new_manual_steering
    set_steering_angle(pred_angle)
    if new_manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if pred_angle < 0 else "right"
        print("turning {} rad {}".format(str(pred_angle),turn))
    

# Constantes para el modelo NVIDIA
MODEL_PATH = r'C:\Users\leduc\Documents\ITESM\MNA\Navegacion\MR4010ProyectoFinal2025\model.h5'
INPUT_WIDTH, INPUT_HEIGHT = 200, 66
FONT = cv2.FONT_HERSHEY_SIMPLEX

def preprocess(img):
    resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    normalized = resized / 127.5 - 1.0
    return normalized.reshape((1, INPUT_HEIGHT, INPUT_WIDTH, 3))

# Detección de objetos usando yolo
def detect_objects_yolo(image):
    # Convertir la imagen de BGRA a BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # Realizar la detección
    results = model2(image_rgb)[0]

    annotated = image_rgb.copy()
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box.int().tolist()
        label = model2.names[int(cls)]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            annotated, text, (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0,255,0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    return annotated
        

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    # Lidar
    lidar = robot.getLidar('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # Create camera instance
    camera2 = robot.getDevice("camera_2")
    camera2.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")
    display_Parameters = Display("display_Parameters")
    
    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    # Carga del modelo (al inicio, una sola vez)
    model = load_model(MODEL_PATH)


    while robot.step() != -1:
        
        #lidar
        range_image = lidar.getRangeImage()
        print("{}".format(range_image[0:5]))
        lidar_dist=round(range_image[0],4)
        lidar_summary = f"Avg: {range_image[0]:.2f} m"
        print(lidar_summary)

        # Get image from camera
        image = get_image(camera)
        image2 = get_image(camera2)
        image_disp=get_image_disp(camera)

        image=region_of_interest(image)
        #vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        proc = preprocess(image)
        #   2) Obtener predicción de ángulo
        pred_angle = float(model.predict(proc, verbose=0)[0])

        # cambia el ángulo del volante
        if pred_angle == 0: #straight
            print("derecha la flecha")
        elif pred_angle > 0: #down
            change_steer_angle_new(pred_angle)
            print("derecha")
        elif pred_angle < 0: #left
            change_steer_angle_new(pred_angle)
            print("izquierda")
        
        # Process and display image 
        grey_image = greyscale_cv2(image_disp)
        edges = detect_edges(grey_image)
        edges = region_of_interestH(edges)
        lines = detect_lines(edges)
        line_overlay = draw_lines(image, lines)

        yolo_image = detect_objects_yolo(image_disp)
        #print(f"yolo_image: {yolo_image}")
        display_image_test(display_img, yolo_image)
        #display_image(display_img_2, vis)
        
        speed_drive=set_speed(lidar_dist)
        print(speed_drive)

        print(f"Pred steering: {pred_angle:.3f} rad")
        #cv2.putText(image, text, (10, 30), FONT, 0.8, (0, 255, 0), 2)
        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

        #Obtenemos un promedio usando el modelo como un 80% del modelo de NVIDIAy con 20% de Hough para ayudarlo a mantenerse en la línea
        try:
            anguloH=set_steering_angle_from_hough_lines(lines)
            angulo_prom=(0.2*anguloH+0.8*pred_angle)
        except:
            angulo_prom=0
            anguloH=0   
        info_overlay = draw_info_overlay(line_overlay, lidar_dist, angulo_prom, speed_drive)
        display_image_test(display_Parameters, info_overlay)

        #update angle and speed
        driver.setSteeringAngle(angulo_prom)
        driver.setCruisingSpeed(speed_drive)


if __name__ == "__main__":
    main()