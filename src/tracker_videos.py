import cv2
import numpy as np
import os

def points_to_tennis_score(points):
    """
    Convierte la cantidad de puntos (0,1,2,3,4,...) 
    a la nomenclatura básica del tenis.
    0 -> Love
    1 -> 15
    2 -> 30
    3 -> 40
    >=4 -> Game
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
        return "Game"

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

    # Covarianza de ruido de proceso (4x4).
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Covarianza de ruido de medida (2x2).
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    # Matriz de error de estimación posterior inicial (4x4).
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman

def process_stream(cap, seconds_absence=1):
    """
    Recibe un objeto VideoCapture (sea webcam o un video),
    y procesa el stream cuadro a cuadro con la lógica de
    detección de pelota y conteo de puntos tipo tenis,
    utilizando además un filtro de Kalman para el seguimiento.
    """

    # Intentamos obtener FPS; si no se puede, forzamos a 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # Cuántos frames consecutivos de ausencia necesitamos para sumar un punto
    absence_frames_threshold = int(seconds_absence * fps)

    # Rango de color HSV para detectar el verde (ajústalo a tu pelota/entorno)
    lower_green = np.array([29, 86, 6], dtype="uint8")
    upper_green = np.array([64, 255, 255], dtype="uint8")

    # Variables para conteo de puntos y juegos
    points = 0
    games = 0
    absent_frames = 0
    already_counted = False

    # Inicializamos el filtro de Kalman
    kalman = init_kalman_filter()

    # Inicializamos el estado (x, y, vx, vy) con valores arbitrarios.
    _, first_frame = cap.read()
    if first_frame is None:
        print("No se pudo leer el video/cámara.")
        return
    height, width, _ = first_frame.shape
    kalman.statePost = np.array([width//2, height//2, 0, 0], dtype=np.float32)

    # Para no perder ese primer frame, volvemos a poner el buffer en el 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Para dibujar el bounding box del tracker, 
    # usamos (w, h) iniciales y los iremos actualizando 
    # cuando detectemos la pelota por color.
    last_w, last_h = 60, 60  # Un tamaño de caja inicial "arbitrario"

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fin del video o error de lectura
            break

        # Predicción de la posición con Kalman
        predicted = kalman.predict()
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        # --- Procesamiento para detectar pelota ---
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

                # Dibujamos la pelota en la imagen (detección por color)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Bounding box de la Detección (NO del Kalman)
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
                cv2.rectangle(frame,
                              (x_rect, y_rect),
                              (x_rect + w_rect, y_rect + h_rect),
                              (0, 255, 0), 2)  # Verde para la detección por color

                ball_in_current_frame = True
                detected_center = (center_x, center_y)

                # Actualizamos last_w, last_h con la última detección real
                last_w, last_h = w_rect, h_rect

        # --- Lógica de Kalman: corrección si se detecta la pelota ---
        if ball_in_current_frame and detected_center is not None:
            measurement = np.array([[np.float32(detected_center[0])],
                                    [np.float32(detected_center[1])]])
            kalman.correct(measurement)
            # Tras la corrección, podemos obtener la posición refinada
            ref_x = int(kalman.statePost[0])
            ref_y = int(kalman.statePost[1])
        else:
            # Si no hay pelota, usamos la predicción
            ref_x, ref_y = pred_x, pred_y

        # --- Dibujamos la posición estimada por Kalman (círculo azul) ---
        cv2.circle(frame, (ref_x, ref_y), 5, (255, 0, 0), -1)

        # --- Bounding box del TRACKER (Kalman) ---
        # Usamos la última detección (last_w, last_h) y la centramos en [ref_x, ref_y].
        x_bb = int(ref_x - last_w // 2)
        y_bb = int(ref_y - last_h // 2)
        cv2.rectangle(frame,
                      (x_bb, y_bb),
                      (x_bb + last_w, y_bb + last_h),
                      (255, 0, 255), 2)  # Morado para la caja del Kalman

        # --- Lógica de ausencia/presencia de la pelota para el conteo ---
        if ball_in_current_frame:
            absent_frames = 0
            already_counted = False
        else:
            absent_frames += 1
            if (not already_counted) and (absent_frames >= absence_frames_threshold):
                points += 1
                already_counted = True

                # Si llega a 4, significa que ganó un Game
                if points >= 4:
                    games += 1
                    points = 0

        # --- Convertimos puntos numéricos a notación de tenis ---
        current_score_text = points_to_tennis_score(points)
        if current_score_text == "Game":
            # Hemos manejado "Game" reseteando 'points',
            # así que si points >=4, en pantalla se volverá a ver "Love".
            current_score_text = "Love"

        # --- Mostramos el marcador ---
        display_text = f"Games: {games} | Score: {current_score_text}"
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Mostramos el frame en pantalla
        cv2.imshow("Tennis Ball Tracker with Kalman + BBoxes", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # =================================
    # Cambia esta variable a tu gusto:
    # True  -> Usar webcam en vivo
    # False -> Procesar lista de videos
    # =================================
    use_webcam = False

    if use_webcam:
        # --- MODO WEBCAM ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la webcam.")
            return

        process_stream(cap, seconds_absence=1)

    else:
        # --- MODO LISTA DE VIDEOS ---
        input_videos = [
            "Lab_Project/data/videos/video1.mp4",
            "Lab_Project/data/videos/video2.mp4",
            "Lab_Project/data/videos/video3.mp4"
        ]

        for video_path in input_videos:
            print(f"Procesando video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"No se pudo abrir el video {video_path}")
                continue

            process_stream(cap, seconds_absence=1)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
