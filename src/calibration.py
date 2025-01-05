from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os


# FUNCIONES AUXILIARES

def load_images(filenames: List) -> List:
    """
    Carga una lista de imágenes desde una lista de rutas de archivos.
    
    Args:
        filenames (List): Lista de rutas a archivos de imágenes.
        
    Returns:
        List: Lista de imágenes cargadas.
    """
    return [imageio.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    """
    Genera los puntos de un patrón de ajedrez en el plano 3D (z=0).
    
    Args:
        chessboard_shape (tuple): Dimensiones del patrón (filas, columnas).
        dx (float): Tamaño de cada cuadrado en el eje x.
        dy (float): Tamaño de cada cuadrado en el eje y.
        
    Returns:
        np.array: Array de puntos 3D del patrón.
    """
    rows, cols = chessboard_shape
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * [dx, dy]
    return objp

def refine_corners(imgs_gray, corners_copy, criteria):
    """
    Refina las esquinas detectadas en las imágenes en escala de grises.
    
    Args:
        imgs_gray (List): Lista de imágenes en escala de grises.
        corners_copy (List): Lista de esquinas detectadas.
        criteria (tuple): Criterios para refinar las esquinas.
        
    Returns:
        List: Esquinas refinadas para cada imagen.
    """
    return [cv2.cornerSubPix(img_gray, cor[1], (7, 7), (-1, -1), criteria) if cor[0] else [] 
            for img_gray, cor in zip(imgs_gray, corners_copy)]

def save_calibrated_images(valid_imgs, valid_corners, patternSize, output_dir):
    """
    Guarda las imágenes con las esquinas detectadas en un directorio.
    
    Args:
        valid_imgs (List): Lista de imágenes válidas.
        valid_corners (List): Lista de esquinas válidas.
        patternSize (tuple): Tamaño del patrón de ajedrez.
        output_dir (str): Ruta del directorio de salida.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img in enumerate(valid_imgs):
        img_copy = img.copy()
        cv2.drawChessboardCorners(img_copy, patternSize, valid_corners[i], True)
        output_path = os.path.join(output_dir, f'right_result_{i + 1}.jpg')
        cv2.imwrite(output_path, img_copy)
    print(f"Imágenes guardadas en {output_dir}")


# LÓGICA PRINCIPAL

if __name__ == "__main__":
    # Obtener una lista de todas las imágenes en el directorio especificado
    print(os.getcwdb())
    image_files = glob.glob('/Users/gonzaloborracherogarcia/ProyectoFinal_CV_REMOTE/data/images/*.jpg')

    # Cargar imágenes desde la lista de archivos
    imgs = load_images(image_files)
    print(f'Loaded {len(imgs)} images')

    # Definir el tamaño del patrón de ajedrez
    patternSize = (7, 7)

    # Detectar esquinas del patrón de ajedrez en cada imagen
    corners = [cv2.findChessboardCorners(img, patternSize, None) for img in imgs]

    # Crear una copia de las esquinas detectadas
    corners_copy = copy.deepcopy(corners)

    # Definir criterios para refinar esquinas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.01)

    # Convertir las imágenes a escala de grises
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    # Refinar las esquinas detectadas
    corners_refined = refine_corners(imgs_gray, corners_copy, criteria)

    # Generar puntos del patrón de ajedrez para todas las imágenes
    chessboard_points = [get_chessboard_points((7, 7), 24, 24) for _ in range(len(corners_refined))]
 
    # Filtrar datos válidos: mantener solo las detecciones adecuadas
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)  # Convertir a array de numpy

    # Filtrar imágenes válidas basadas en esquinas refinadas
    valid_imgs = [img for img, cor in zip(imgs, corners_refined) if cor is not None]

    # Calibrar la cámara usando los puntos detectados y refinados
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, corners_refined, (1280, 720), None, None
    )

    # Obtener los parámetros extrínsecos
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    # Imprimir los resultados de la calibración
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)

    # Guardar las imágenes con las esquinas detectadas
    output_dir = "/Users/gonzaloborracherogarcia/ProyectoFinal_CV_REMOTE/data/images/calibration_results"
    save_calibrated_images(valid_imgs, valid_corners, patternSize, output_dir)
