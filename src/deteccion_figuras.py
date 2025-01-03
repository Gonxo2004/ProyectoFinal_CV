import os
import cv2
import numpy as np

# =======================
# FUNCIONES AUXILIARES
# =======================

def detect_shape(contour):
    """Determina la forma de un contorno (triángulo, cuadrado, círculo)."""
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
    """Segmenta un color específico en la imagen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, segmented

def process_frame_for_shapes(frame, min_area=1000):
    """Procesa un fotograma para detectar formas geométricas y colores específicos."""
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
            cv2.putText(frame, f"{shape} ({int(area)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, detected_shapes

def detect_colored_shapes(frame, color_ranges, min_area=1000):
    detected_colored_shapes = []

    for color_name, (lower, upper) in color_ranges.items():
        mask, segmented = segment_color(frame, lower, upper)
        _, detected_shapes = process_frame_for_shapes(segmented, min_area)

        for shape, area in detected_shapes:
            detected_colored_shapes.append(f"{shape} {color_name}")

    return detected_colored_shapes

# =======================
# PROCESAMIENTO EN TIEMPO REAL
# =======================

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        exit()

    min_area = 10000

    color_ranges = {
        "rojo": ((0, 100, 100), (10, 255, 255)),
        "amarillo": ((20, 100, 100), (30, 255, 255)),
        "azul": ((100, 150, 0), (140, 255, 255)),
    }

    unlock_sequence = ["Cuadrado rojo", "Triangulo amarillo", "Circulo azul"]
    detected_sequence = []
    state_machine = {item: False for item in unlock_sequence}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma.")
            break

        detected_colored_shapes = detect_colored_shapes(frame, color_ranges, min_area)

        y_offset = 30
        for colored_shape in detected_colored_shapes:
            if state_machine.get(colored_shape, False):
                continue

            if len(detected_sequence) < len(unlock_sequence) and colored_shape == unlock_sequence[len(detected_sequence)]:
                detected_sequence.append(colored_shape)
                state_machine[colored_shape] = True
                if detected_sequence == unlock_sequence:
                    cv2.putText(frame, "Desbloqueo exitoso!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    detected_sequence = []
            
            cv2.putText(frame, f"Detectado: {colored_shape}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 40

        cv2.putText(frame, f"Secuencia: {' -> '.join(detected_sequence)}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Deteccion de Formas y Colores", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
