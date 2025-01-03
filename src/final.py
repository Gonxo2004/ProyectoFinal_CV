import cv2
import numpy as np

# =============== 1) FUNCIONES DE DETECCIÓN DE FORMAS Y COLORES ===============

def detect_shape(contour):
    """Determina la forma de un contorno (triángulo, cuadrado, círculo, etc.)."""
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 3:
        return "Triangulo"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            return "Cuadrado"
        else:
            return "Rectangulo"
    elif len(approx) > 4:
        return "Circulo"
    else:
        return "Otro"

def segment_color(frame, lower_bound, upper_bound):
    """
    Segmenta un color específico en la imagen (devuelve mask y la imagen segmentada).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, segmented

def process_frame_for_shapes(frame, min_area=1000):
    """
    Detecta contornos grandes y determina la forma (lo dibuja en frame).
    Retorna:
      - frame con contornos dibujados
      - lista de (shape, area)
    """
    detected_shapes = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            shape = detect_shape(contour)
            detected_shapes.append((shape, area))
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, f"{shape} ({int(area)})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    return frame, detected_shapes

def detect_colored_shapes(frame, color_ranges, min_area=1000):
    """
    Detecta formas en cada rango de color y retorna lista de strings 
    del tipo 'Triangulo amarillo', 'Cuadrado rojo', etc.
    """
    detected_colored_shapes = []

    for color_name, (lower, upper) in color_ranges.items():
        mask, segmented = segment_color(frame, lower, upper)
        _, shapes = process_frame_for_shapes(segmented, min_area)

        for shape, area in shapes:
            detected_colored_shapes.append(f"{shape} {color_name}")

    return detected_colored_shapes

# =============== 2) LÓGICA PARA EL MARCADOR DE TENIS (SIN KALMAN) ===============

def points_to_tennis_score(points):
    """
    Convierte la cantidad de puntos (0,1,2,3,4,...) 
    a la nomenclatura básica del tenis:
    0 -> Love, 1 -> 15, 2 -> 30, 3 -> 40, >=4 -> Game.
    """
    if points == 0:
        return "Love"
    elif points == 1:
        return "15"
    elif points == 2:
        return "30"
    elif points == 3:
        return "40"
    else:
        # Si llega a 4 o más, consideramos que ganó un Game
        return "Game"

# =============== 3) KALMAN FILTER PARA LA PELOTA DE TENIS ===============

def init_kalman_filter():
    """
    Inicializa un filtro de Kalman sencillo para estimar:
    estado = [x, y, vx, vy]
    medida = [x, y]
    """
    kalman = cv2.KalmanFilter(4, 2)  # 4 estados, 2 mediciones

    # Matriz de transición de estados (4x4).
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Matriz de observación (2x4): medimos x, y.
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Covarianza de ruido de proceso.
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Covarianza de ruido de medida.
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    # Matriz de error de estimación posterior inicial.
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman

# =============== 4) FUNCIÓN PRINCIPAL QUE UNE TODO ===============

def main():
    # --- Apertura de la cámara ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    # ============ PARÁMETROS PARA LA DETECCIÓN DE FIGURAS/COLORES ============
    min_area = 10000
    color_ranges = {
        "rojo": ((0, 100, 100), (10, 255, 255)),
        "amarillo": ((20, 100, 100), (30, 255, 255)),
        "azul": ((100, 150, 0), (140, 255, 255)),
    }
    unlock_sequence = ["Cuadrado rojo", "Triangulo amarillo", "Circulo azul"]
    detected_sequence = []  
    state_machine = {item: False for item in unlock_sequence}

    # ============ PARÁMETROS PARA TRACKER DE PELOTA (KALMAN) ============
    unlocked = False
    points = 0
    games = 0
    absent_frames = 0
    already_counted = False
    
    # Intentamos obtener FPS; si no funciona, suponemos 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    seconds_absence = 1
    absence_frames_threshold = int(seconds_absence * fps)

    # Rango de color para la pelota verde
    lower_green = np.array([29, 86, 6], dtype="uint8")
    upper_green = np.array([64, 255, 255], dtype="uint8")

    # --- Inicializamos Kalman Filter (aunque se use solo al desbloquear) ---
    kalman = init_kalman_filter()
    last_w, last_h = 60, 60  # Tamaño inicial de la bounding box del tracker

    # --- Leemos el primer frame para ajustar el estado inicial del Kalman ---
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("No se pudo leer la cámara en el primer frame.")
        cap.release()
        cv2.destroyAllWindows()
        return

    height, width, _ = first_frame.shape
    # Asignamos [x, y, vx, vy] en el centro de la imagen
    kalman.statePost = np.array([width//2, height//2, 0, 0], dtype=np.float32)

    # Regresamos el "puntero" de la cámara al inicio (por precaución)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma.")
            break

        # 1) DETECCIÓN DE FIGURAS DE COLOR (SI AÚN NO ESTÁ DESBLOQUEADO)
        if not unlocked:
            detected_colored_shapes = detect_colored_shapes(frame, color_ranges, min_area)

            # Dibujamos la secuencia actual en pantalla
            cv2.putText(frame, f"Secuencia: {' -> '.join(detected_sequence)}", 
                        (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (255, 255, 255), 
                        2)

            # Procesamos las figuras detectadas y vemos si encajan en la secuencia
            y_offset = 30
            for colored_shape in detected_colored_shapes:
                # Si la figura ya fue detectada, no la repetimos
                if state_machine.get(colored_shape, False):
                    continue

                # Verificamos si coincide con el siguiente paso de la secuencia
                if (len(detected_sequence) < len(unlock_sequence) and
                    colored_shape == unlock_sequence[len(detected_sequence)]):
                    detected_sequence.append(colored_shape)
                    state_machine[colored_shape] = True

                    # ¿La secuencia está completa?
                    if detected_sequence == unlock_sequence:
                        cv2.putText(frame, 
                                    "Desbloqueo exitoso!", 
                                    (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (0, 255, 0), 
                                    3)
                        unlocked = True  # ¡Activamos el tracker de tenis!
                
                # Mostrar texto de lo que detectó en este frame
                cv2.putText(frame, f"Detectado: {colored_shape}", 
                            (30, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 255), 
                            2)
                y_offset += 40
        
        # 2) TRACKER DE PELOTA (KALMAN) SI ESTÁ DESBLOQUEADO
        if unlocked:
            # 2.1) Predicción del Kalman
            predicted = kalman.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])

            # 2.2) Detección de la pelota verde por color
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ball_in_current_frame = False
            detected_center = None

            if contours:
                # Tomamos el contorno más grande
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 10:
                    M = cv2.moments(c)
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # DIBUJAMOS bounding box de la detección (verde)
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
                    cv2.rectangle(frame, 
                                  (x_rect, y_rect), 
                                  (x_rect + w_rect, y_rect + h_rect), 
                                  (0, 255, 0), 2)
                    
                    # DIBUJAMOS la pelota (círculo amarillo + punto rojo)
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    ball_in_current_frame = True
                    detected_center = (center_x, center_y)

                    # Actualizamos el tamaño de la bbox para el tracker
                    last_w, last_h = w_rect, h_rect

            # 2.3) Corrección del Kalman si se detectó la pelota
            if ball_in_current_frame and detected_center is not None:
                measurement = np.array([[np.float32(detected_center[0])],
                                        [np.float32(detected_center[1])]])
                kalman.correct(measurement)
                ref_x = int(kalman.statePost[0])
                ref_y = int(kalman.statePost[1])
            else:
                # Si no hay pelota, usamos la predicción
                ref_x, ref_y = pred_x, pred_y

            # 2.4) Dibujamos la posición estimada por Kalman (círculo azul)
            cv2.circle(frame, (ref_x, ref_y), 5, (255, 0, 0), -1)

            # 2.5) Bounding box del TRACKER (morado)
            x_bb = int(ref_x - last_w // 2)
            y_bb = int(ref_y - last_h // 2)
            cv2.rectangle(frame,
                          (x_bb, y_bb),
                          (x_bb + last_w, y_bb + last_h),
                          (255, 0, 255), 2)

            # 2.6) Lógica de ausencia para conteo de puntos
            if ball_in_current_frame:
                absent_frames = 0
                already_counted = False
            else:
                absent_frames += 1
                if (not already_counted) and (absent_frames >= absence_frames_threshold):
                    points += 1
                    already_counted = True

                    # Si llega a 4, es "Game"
                    if points >= 4:
                        games += 1
                        points = 0

            # 2.7) Mostramos el marcador tipo tenis
            current_score_text = points_to_tennis_score(points)
            if current_score_text == "Game":
                # Si era "Game", se reseteó points, así que mostramos "Love"
                current_score_text = "Love"

            display_text = f"Games: {games} | Score: {current_score_text}"
            cv2.putText(frame, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3) MOSTRAR EL FRAME FINAL
        cv2.imshow("Fusion: Deteccion de Figuras + Tracker de Tenis (Kalman)", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
