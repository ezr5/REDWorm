# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:25:59 2024

@author: Eva
"""
#librerías
import os
import csv
from skimage import morphology
import matplotlib.pyplot as plt
import multipagetiff as mtif
import numpy as np 
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skan import csr
from skan import Skeleton, summarize
from scipy.stats import skew, kurtosis 
import tifffile
import webbrowser
from pathlib import Path
import  tkinter as tk
from tkinter import Tk, ttk, Canvas, Button, PhotoImage, filedialog


#------------------------------------------CÓDIGO ---------------------------------------------------------------

# Función para obtener el esqueleto de una imagen pasada como parámetro
def obtain_skeleton(image):
    # Aplicar umbralización + invertir
    enhanced_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)  # Aumentar el contraste y convertir a 8 bits
    thresh = threshold_otsu(enhanced_image)  # Calcular umbral óptimo
    binary_image = enhanced_image > thresh  # aplicar umbral creando imagen binaria
    inverted_image = np.logical_not(binary_image)  # Invertir la imagen binaria
    # Eliminar artefactos conectados al borde de la imagen
    cleared = clear_border(inverted_image)  # Limpiar bordes conectados
    closed = morphology.closing(cleared, disk(1))  # operación morfológica de cierre con forma de diamante 
    dilation = morphology.binary_dilation(closed, disk(1))  # Dilatar la imagen con forma de diamente
    # Eliminar todos los elementos encontrados, menos el gusano(el más grande)
    labeled_image = label(dilation)  # Volver a etiquetar la imagen
    label_sizes = np.bincount(labeled_image.ravel())  # Contar el tamaño de las etiquetas
    largest_label = label_sizes[1:].argmax() + 1  # Encontrar la etiqueta más grande
    largest_component = labeled_image == largest_label  # Crear una máscara de la etiqueta más grande
    # Obtener el esqueleto 
    skeleton = morphology.skeletonize(largest_component)  # Crear el esqueleto
    dilation_skeleton = morphology.binary_dilation(skeleton)  # Dilatar el esqueleto
    skeleton2 = morphology.skeletonize(dilation_skeleton)  # Crear el esqueleto final
    # Eliminar pequeñas ramas del esqueleto
    graph_class = csr.Skeleton(skeleton2)  # Crear objeto para analizar el esqueleto
    stats = csr.summarize(graph_class)  # Resumir estadísticas del esqueleto
    skeleton_size = np.sum(skeleton2) #obtener el tamaño total del esqueleto
    # Ajustar el umbral de eliminación de ramas en función del tamaño del esqueleto
    thres_min_size = round(skeleton_size * 0.3,1)  #el valor del umbral es el 0.3 del tamaño del esqueleto
    for i in range(stats.shape[0]):  # Iterar sobre las ramas
        #si la longitud de la rama es menor que el umbral y es del tipo 1
        if (round(stats.loc[i, 'branch-distance'],1) <= thres_min_size and stats.loc[i, 'branch-type'] == 1):
            # Obtener las coordenadas de indexación de NumPy, ignorando los puntos finales
            integer_coords = tuple(
                graph_class.path_coordinates(i)[1:-1].T.astype(int)
            )
            # Eliminar la rama
            skeleton2[integer_coords] = 0
    skclean = morphology.remove_small_objects(skeleton2, min_size=2, connectivity=2)  # Eliminar objetos pequeños
    return skclean  # Devolver el esqueleto limpio


# funcion para calcular la pendiente de la línea que une la cabeza y la cola (eje Y)
def calculate_slope_line(head_coords, tail_coords):
    if (head_coords[0] != tail_coords[0]) and (head_coords[1] != tail_coords[1]):  # Evitar división por cero
        slope_given_line = (head_coords[1] - tail_coords[1]) / (head_coords[0] - tail_coords[0]) #obtener pendiente de la recta
    elif head_coords[0] == tail_coords[0]:  # Si la línea es vertical
        slope_given_line = np.inf #la pendiente es infinito
    else:  # Si la línea es horizontal
        slope_given_line = 0 #la pendiente es 0
    
    return slope_given_line

# funcion para calcular los puntos que conforman la línea perpendicular (eje X)
def calculate_perpendicular_line(slope_line,point):
    delta = 50  # Distancia desde el punto medio para determinar los puntos de la linea perpendicular
    if slope_line != 0 and slope_line != np.inf:  # Cuando la línea no es vertical ni horizontal
        #se obtienen los puntos aplicando las ecuaciones de las rectas
        slope_perpendicular_line = -1 / slope_line 
        point1 = (point[0] + delta, point[1] + delta * slope_perpendicular_line)
        point2 = (point[0] - delta, point[1] - delta * slope_perpendicular_line)
    elif slope_line == np.inf:  # Si la pendiente es vertical
        #se obtienen los puntos sumando o restando delta a lo largo x
        point1 = (point[0] + delta, point[1])
        point2 = (point[0] - delta, point[1])
    else:  # Si la pendiente es horizontal
        #se obtienen los puntos sumando o restando delta a lo largo y
        point1 = (point[0], point[1] + delta)
        point2 = (point[0], point[1] - delta)
    
    return [point1,point2]

# Función para calcular el nuevo origen de coordenadas
def plot_coordinate_axes(skeleton):
     endpoints = end_points(skeleton)
     if len(endpoints) <= 1:
        return detect_coil(skeleton)
     else:
        head_coords = endpoints[0]  # Coordenadas de la cabeza coincide con la primera tupla
        tail_coords = endpoints[1]  # Coordenadas de la cola coincide con la segunda tupla

        # Calcular la pendiente de la línea que une la cabeza y la cola (eje Y)
        slope_given_line = calculate_slope_line(head_coords, tail_coords)
        # Calcular el punto medio entre la cabeza y la cola
        mid_point = ((head_coords[0] + tail_coords[0]) / 2, (head_coords[1] + tail_coords[1]) / 2)
        #obtener los puntos de la línea perperpendicular (eje X) a partir del eje Y
        points_perpendicular_line = calculate_perpendicular_line(slope_given_line, mid_point)
        point1 = points_perpendicular_line[0]
        point2 = points_perpendicular_line[1]
        # Encontrar la intersección entre la línea perpendicular y la línea que une la cabeza y la cola
        x_intersect, y_intersect = lineLineIntersection(head_coords, tail_coords, point1, point2)
        intersection = (x_intersect, y_intersect)
        return intersection

#obtener distancias entre la recta cabeza-cola y los puntos de intersección con el
def plot_lines_along_line(skeleton):
    endpoints = end_points(skeleton) #obtener endpoints
    if len(endpoints) <= 1: #si se tiene menos de un endpoint se tratará de un coil
        return detect_coil(skeleton) #se llama a la función que detecta los coils
    else: #si no esta en coil el gusano
        head_coords =endpoints[0] #se obtienen coordenadas de la cabeza
        tail_coords = endpoints[1] #se obtienen coordenadas de la cola
        
        #calcular el número de líneas que hay que trazar
        #para ello se calculan los el número de segmentos que van a delimitar las líneas
        distancia = distance(head_coords,tail_coords) #se obtiene la distancia entre cabeza y cola
        num_segments = max(int(distancia / 10), 1) #se calculan los segmentos
        if num_segments < 4: #si hay menos de 4
            num_segments = 4 #como minimo 4 segmentos
        
        # Calcular las coordenadas de los puntos a lo largo de la línea que une la cabeza y la cola
        points_on_line = [] #se inicializa el array para guardar los puntos
        for i in range(1, num_segments + 1):  # Iterar sobre el número de segmentos deseados
            factor = i / (num_segments + 1)  # Calcular el factor de interpolación entre la cabeza y la cola
            # Calcular la coordenada x del punto interpolado usando una ponderación entre las coordenadas x de la cabeza y la cola
            x_point = head_coords[0] * factor + tail_coords[0] * (1 - factor)
            # Calcular la coordenada y del punto interpolado usando una ponderación entre las coordenadas y de la cabeza y la cola
            y_point = head_coords[1] * factor + tail_coords[1] * (1 - factor)
            # Agregar las coordenadas del punto interpolado a la lista de puntos a lo largo de la línea
            points_on_line.append((x_point, y_point))
        
        #calcular las distancias de cada línea
        distancias = [] #se inicializa el array para guardar las distancias
        # Calcular la pendiente de la línea que une la cabeza y la cola (eje Y)
        slope_given_line = calculate_slope_line(head_coords, tail_coords)
        #para cada uno de los puntos a lo largo del eje Y
        for point in points_on_line:
            #obtener los puntos de la línea perperpendicular (eje X) a partir del eje Y
            points_perpendicular_line = calculate_perpendicular_line(slope_given_line, point)
            point1 = points_perpendicular_line[0]
            point2 = points_perpendicular_line[1]
            point_ske = find_intersection(point1,point2,skeleton) #obtener intersección con el esqueleto
            dist_point_ske = distance(point,point_ske) #obtener distancia entre el punto y el esqueleto
            if not signo_dist(point_ske, point,slope_given_line): #si la distancia es negativa
                dist_point_ske = -dist_point_ske #se pone el signo negativo
            distancias.append(dist_point_ske) #se añade la distancia al array de distancias
        return distancias

#función para determinar el signo de la distancia en función 
#de donde se situe el point1(punto en el cuerpo del esqueleto) respecto
#al nuevo origen de coordenadas
def signo_dist(point1, point_line, slope_given_line):
    # Calcular la diferencia entre las coordenadas x e y de los puntos
    dx = round(point1[0],1) - round(point_line[0],1)  # Diferencia en coordenadas x redondeada a un decimal
    dy = round(point1[1],1) - round(point_line[1],1)  # Diferencia en coordenadas y redondeada a un decimal
    
    # Verificar el signo de la distancia basándose en la pendiente de la línea
    if slope_given_line == np.inf:  # Si la línea es vertical
        return dx >= 0  # Devolver True si la dx es positiva, False de lo contrario
    elif slope_given_line == 0:  # Si la línea es horizontal
        return dy >= 0  # Devolver True si la dy es positiva, False de lo contrario
    else:  # Si la pendiente es diferente de infinito y de cero (es decir, la línea no es vertical ni horizontal)
        if slope_given_line > 0:  # Si la pendiente es positiva
            return dx >= 0  # Devolver True si la dx es positiva, False de lo contrario
        else:  # Si la pendiente es negativa
            return dy >= 0  # Devolver True si la dy es positiva, False de lo contrario

#funcion que encuentra las coordenadas de intersección entre dos rectas
#formadas cada una por las coordenadas de inicio y fin de la misma
#recta 1 = (Ax,Ay), (Bx,By)
#recta 2 = (Cx,Cy), (Dx,Dy)
def lineLineIntersection(A, B, C, D):
    # Representación de la línea AB como a1x + b1y = c1
    a1 = B[1] - A[1]  # Coeficiente a de la ecuación de la línea AB
    b1 = A[0] - B[0]  # Coeficiente b de la ecuación de la línea AB
    c1 = a1 * A[0] + b1 * A[1]  # Coeficiente c de la ecuación de la línea AB
 
    # Representación de la línea CD como a2x + b2y = c2
    a2 = D[1] - C[1]  # Coeficiente a de la ecuación de la línea CD
    b2 = C[0] - D[0]  # Coeficiente b de la ecuación de la línea CD
    c2 = a2 * C[0] + b2 * C[1]  # Coeficiente c de la ecuación de la línea CD
 
    # Calcular el determinante
    determinant = a1 * b2 - a2 * b1
    
    # Calcular las coordenadas del punto de intersección
    x = (b2 * c1 - b1 * c2) / determinant  # Coordenada x del punto de intersección
    y = (a1 * c2 - a2 * c1) / determinant  # Coordenada y del punto de intersección

    return (x, y)  # Devolver las coordenadas del punto de intersección

#funcion que calcula la distancia euclidiana entre dos puntos
def distance(point1, point2):
    #se obtienen las componentes x e y de cada punto
    x1, y1 = point1 
    x2, y2 = point2
    #se realiza la diferencia de x e y
    dx = x2 - x1
    dy = y2 - y1
    # se calcula la distancia euclidiana
    distancia = np.sqrt(dx**2 + dy**2)
    return round(distancia,2) #se devuelve el valor de la distancia redondeada a 2 decimales

#función que calcula la longitud del esqueleto del gusano
def skeleton_length(image_skeleton):
    branch_data = summarize(Skeleton(image_skeleton)) # se obtiene un resumen de las estadísticas del esqueleto
    longitud_total = sum(branch_data['branch-distance'])# se suma las longitudes de todas las posibles ramas
    return round(longitud_total,2) #se devuelve la longitud redondeada a 2 decimales

#función que obtiene las coordenadas de los puntos finales del esqueleto
def end_points(skeleton):
    (rows, cols) = np.nonzero(skeleton) # Encontrar las ubicaciones de filas y columnas que no son cero
    skel_coords = []  # Inicializar una lista vacía de coordenadas
    # Para cada píxel no cero...
    for (r, c) in zip(rows, cols):
        # Extraer un vecindario 8-conectado
        (col_neigh, row_neigh) = np.meshgrid(np.array([c-1, c, c+1]), np.array([r-1, r, r+1]))
        # Convertir en enteros para indexar en la imagen
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        # Convertir en un solo arreglo 1D y verificar las ubicaciones no cero
        pix_neighbourhood = skeleton[row_neigh, col_neigh].ravel() != 0
        if np.sum(pix_neighbourhood) == 2:  # Verificar si el píxel actual no es cero y al menos uno de sus vecinos es parte del esqueleto
            skel_coords.append((c, r)) # Agregar las coordenadas a la lista
    
    return skel_coords  # Devolver las coordenadas de los puntos finales del esqueleto

#función que encuentra el punto de intersección más cercano entre una línea definida 
#por dos puntos (point_start y point_end) y un esqueleto
def find_intersection(point_start, point_end, skeleton):
    # Encontrar píxeles no cero en el esqueleto
    (filas, columnas) = np.nonzero(skeleton)
    dx = point_end[0] - point_start[0] #calcular la diferencia en x de ambas lineas
    dy = point_end[1] - point_start[1] #calcular la diferencia en y de ambas lineas
    
    distancias = [] #array para almacenar todas las distancias encontradas
    if dx == 0: #si la linea es vertical
        for x in columnas: #se itera sobre las columnas
            distancia = abs(point_start[0] - x) #se calcula la distancia horizontal entre el pixel del esqueleto y la línea
            distancias.append(distancia) #se añade al array
    else: # si la linea no es vertical
        m = dy / dx  # se calcula la pendiente
        b = point_start[1] - m * point_start[0]  #se calcula la intersección en y
        for (x, y) in zip(columnas, filas):
            distancia = abs(y - (m * x + b)) # se calcula la distancia perpendicular entre cada píxel del esqueleto y la línea
            distancias.append(distancia) #se añade al array
    
    valor_mas_pequeno = min(distancias) #se encuentra el valor más pequeño de la distancia
    indice = distancias.index(valor_mas_pequeno) #se obtiene su índice
    return (columnas[indice], filas[indice]) # se devuelve la coordenada del punto de intersección más cercano

#función para detectar coil
def detect_coil(skeleton):
    points = end_points(skeleton)  # Obtener los puntos finales del esqueleto
    giro_type = "" # Variable para almacenar el tipo de giro
    if len(points) <= 1: # Verificar si hay menos de dos puntos finales en el esqueleto
        giro_type = "Coil" # se considera como un "Coil"
    else:# Si hay al menos dos puntos finales en el esqueleto
        head = points[0] #se obtiene la cabeza del gusano
        center = plot_coordinate_axes(skeleton) #se obtiene el origen del nuevo sistema de coordenadas
        distancia = distance(head, center) # Calcular la distancia entre el origen y la cabeza del gusano
        if distancia < 15:# Verificar si la distancia es menor que 15
            giro_type = "Coil" #se considera como un "Coil" 
        else:# Si la distancia es mayor o igual a 15
            giro_type = "None" #No se considera "Coil"
    return giro_type #se devuelve si hay giro "coil" o "none"

#función que devuelve el array que se le pasa como parámetro en valor absoluto
def valor_absoluto_array(array):
    valores_absolutos = [abs(numero) for numero in array]
    return valores_absolutos #devuelve el array con valores absolutos

#funcion que calcula la media de los valores del array que se le pasa como parámetro
def calcular_media(array):
    return round(sum(array) / len(array),2) #devuelve el valor redondeado a 2 decimales

#función que calcula la media de los valores del array por archivo, es decir,
# tiene en cuenta que cada archivo del gusano esta compuesto por 1000 páginas
def calcular_media_gusano(array):
    return round(sum(array) / 1000,2)  #devuelve el valor redondeado a 2 decimales

#funcion que calcula el ratio de la cabeza a la cola del gusano
def ratio_tail_head(length,distance):
    if distance==0: #si se trata de coil
        return 0 #el ratio es 0
    else: #si no esta realizando coil se obtiene el ratio
        return round(length/distance,2) #ratio = longitud esqueleto / distancia cabeza-cola

#función que estudia el signo de los números del array
def check_numbers(array):
    all_positive = all(num >= 0 for num in array) #devuelve True si todos los valores son positivos
    all_negative = all(num < 0 for num in array) #devuelve True si todos los valores son negativos
    
    if all_positive: #si todos los valores son positivos
        return "+" #se devuelve el símbolo +
    elif all_negative: #si todos los valores son negativos
        return "-"  #se devuelve el símbolo -
    else: #si hay valores positivos y negativos
        return "+ y -" #se devuelve "+ y -"

#función que genera un CSV en el "output_path" con los valores de los parámetros por cada frame de cada archivo
#que este situado en el "folder_path"
def obtener_csv(folder_path, output_path):
    # Obtener una lista de todos los archivos TIFF en el directorio que no sean el esqueleto generado
    tiff_files = [file for file in os.listdir(folder_path) if (file.endswith(".tiff") or file.endswith(".tif")) and not 
                   os.path.splitext(os.path.basename(file))[0].endswith("_skeleton")]

    # Crear un archivo CSV para escribir los datos con el nombre de la carpeta donde se encuentre
    csv_file = open(os.path.join(output_path, os.path.basename(folder_path)+"_datos.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)

    # Definir los títulos para el archivo CSV
    titulos = ["ImageName", "Frame", "Distance head-tail", "Distance head-center", "Distance center-tail",
            "Worm length", "Coil (T/F)", "Ratio head-tail", "AVG Distance ABS", "AVG Distance", "Distances",
            "Sign Classification", "Max dist", "Min dist", "Max abs dist", "Min abs dist", "Skewness", "Kurtosis"]

    # Escribir los títulos en la primera fila del archivo CSV
    csv_writer.writerow(titulos)
    # Iterar sobre cada archivo TIFF en la carpeta
    for file_name in tiff_files:
        # Leer el archivo TIFF de la ruta de la carpeta
        image_stack = mtif.read_stack(os.path.join(folder_path, file_name))
        #de cada frame del archivo TIFF...
        for i, image in enumerate(image_stack):
            skeleton = obtain_skeleton(image)  # Obtener el esqueleto de la imagen
            longitud = skeleton_length(skeleton)  # Calcular la longitud del esqueleto
            points = end_points(skeleton)  # Encontrar los puntos finales del esqueleto
            if len(points) <= 1: #si hay menos de dos puntos terminales (el gusano está haciendo coil)
                #las distancias se ponen a 0
                distance_co_tail = 0
                distance_co_center = 0
                distance_ta_center = 0
            else: #si el gusano no esta haciendo "Coil"
                #se obtienen las coordenadas de la cabeza y cola
                head = points[0]
                tail = points[1]
                center = plot_coordinate_axes(skeleton)  # se calcula el origen del nuevo sistema de coordenadas
                distance_co_tail = distance(head, tail)  # se calcula la distancia entre la cabeza y la cola
                distance_co_center = distance(tail, center)  # se calcula la distancia entre la cola y el origen
                distance_ta_center = distance(head, center)  # se calcula la distancia entre la cabeza y el origen

            omega_turns_result = detect_coil(skeleton)  # Detectar coil
            coil = False #se inicializa a false
            if omega_turns_result == "Coil": #si se trata de coil
                coil = True #se cambia a True

            ratio = ratio_tail_head(longitud, distance_co_tail)  # se calcula el ratio entre la cabeza y la cola
            distancias = plot_lines_along_line(skeleton)  # se calcula las distancias a lo largo del eje Y que une cabeza y cola
            if distancias != "Coil": #si no esta realizando Coil el gusano
                distancias_abs = valor_absoluto_array(distancias)  # se calcula los valores absolutos de las distancias
                media_distancia = calcular_media(distancias)  # se calcula la media de las distancias
                media_distancia_abs = calcular_media(distancias_abs)  # se calcula la media de las distancias absolutas
                clasificacion = check_numbers(distancias)  #se obtiene un símbolo en función del signo de las distancias
                max_dist = max(distancias)  # se encuentra la distancia máxima
                max_dist_abs = max(distancias_abs)  # se encuentra la distancia absoluta máxima
                min_dist = min(distancias)  # se encuentra la distancia mínima
                min_dist_abs = min(distancias_abs)  # se encuentra la distancia absoluta mínima
                skewness = round(skew(distancias, axis=0, bias=True), 3)  # se calcula  la asimetría
                kurto = round(kurtosis(distancias, axis=0, bias=True), 3)  # se calcula la curtosis
                
                num_iterations = len(distancias)  # se obtiene el número de iteraciones dependiendo del número de distancias que haya
            # Escribir los datos en el archivo CSV
                for j in range(num_iterations): #se escriben los datos por frame tantas veces como distancias tenga
                    row_data = [file_name, i, distance_co_tail, distance_co_center, distance_ta_center, longitud, coil, ratio,
                                media_distancia_abs, media_distancia, distancias[j], clasificacion, max_dist, min_dist,
                                max_dist_abs, min_dist_abs, skewness, kurto] #donde el valor de las distancias va cambiando
                    csv_writer.writerow(row_data) #se escriben los datos en cada fila del CSV
            else: #si no tiene distancias porque el gusano esta haciendo coil
                #se escriben todos los parámetros menos los correspondientes con el array distancias
                row_data = [file_name, i, distance_co_tail, distance_co_center, distance_ta_center, longitud, coil, ratio]
                csv_writer.writerow(row_data) #se escribe en cada fila del CSV

    # se cierra el archivo CSV
    csv_file.close()
    print("OK")

#función que genera un archivo TIFF del esqueleto del gusano en el "output_path" por
#cada archivo TIFF que se situa en el "folder path"
def create_tiff_skeleton(folder_path, output_path):
     # Obtener una lista de todos los archivos TIFF en el directorio que no sean el esqueleto generado
     tiff_files = [file for file in os.listdir(folder_path) if (file.endswith(".tiff") or file.endswith(".tif")) and not 
                   os.path.splitext(os.path.basename(file))[0].endswith("_skeleton")]
     # se iterar sobre cada archivo TIFF en la carpeta
     for tiff_name in tiff_files:
        # se lee el archivo TIFF de entrada
        image_stack = mtif.read_stack(os.path.join(folder_path, tiff_name))
        skeleton_stack = []  # Inicializar una lista para almacenar los esqueletos de las imágenes
        # Aplicar la función obtain_skeleton a cada frame en la pila de imágenes
        for image in image_stack:
            skeleton = obtain_skeleton(image)  # Obtener el esqueleto de la imagen
            img_8bit = np.clip(np.round(skeleton * 255), 0, 255).astype(np.uint8) #se convierte el esqueleto a 8 bits
            skeleton_stack.append(img_8bit) #se añade al stack
        
        base_name, extension = os.path.splitext(tiff_name) # se obtiene el nombre del archivo y la extensión
        # se define la ruta de salida y se añade la terminación "_skeleton" al archivo creado
        output_file = os.path.join(output_path, base_name+"_skeleton"+extension) 
        tifffile.imwrite(output_file, skeleton_stack, dtype=np.uint8) # se escribir la pila de esqueletos en un archivo TIFF de 8bits

#función que obtiene un CSV en el "output_path" con los valores medios de cada archivo TIFF situado en la "folder_path"
def obtener_csv_media(folder_path, output_path):
    # Obtener una lista de todos los archivos TIFF en el directorio que no sean el esqueleto generado
    tiff_files = [file for file in os.listdir(folder_path) if (file.endswith(".tiff") or file.endswith(".tif")) and not 
                   os.path.splitext(os.path.basename(file))[0].endswith("_skeleton")]
    # Crear un archivo CSV para escribir los datos
    csv_file = open(os.path.join(output_path, os.path.basename(folder_path)+"_datos_medios.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)

    # Definir los títulos para el archivo CSV
    titulos = ["ImageName", "Distance head-tail", "Distance head-center", "Distance center-tail",
            "Worm length","Ratio head-tail", "AVG Distance ABS", "AVG Distance",
            "Max dist", "Min dist", "Max abs dist", "Min abs dist", "Skewness", "Kurtosis"]

    # Escribir los títulos en la primera fila del archivo CSV
    csv_writer.writerow(titulos)
    # Iterar sobre cada archivo TIFF en la carpeta
    for file_name in tiff_files:
        # Inicializar listas para almacenar los datos de cada imagen
        distance_h_t_array = []
        distance_h_c_array = []
        distance_c_t_array = []
        length_array = []
        ratio_array = []
        Avr_abs_dis_array = []
        Avr_dis_array = []
        max_dist_array = []
        min_dist_array = []
        max_abs_dist_array = []
        min_abs_dist_array = []
        ske_array = []
        kurtosis_array = []

        image_stack = mtif.read_stack(os.path.join(folder_path, file_name)) # se lee el archivo TIFF
        #de cada frame del archivo TIFF...
        for i, image in enumerate(image_stack):
            # Procesar la imagen y obtener los datos
            skeleton = obtain_skeleton(image)  # Obtener el esqueleto de la imagen
            longitud = skeleton_length(skeleton)  # Calcular la longitud del esqueleto
            length_array.append(longitud) #se añade al array de longitud

            points = end_points(skeleton)  # Encontrar los puntos finales del esqueleto
            if len(points) > 1: #si hay mas de un endpoint
                #se obtienen las coordenadas de la cabeza y cola
                head = points[0]
                tail = points[1]
                center = plot_coordinate_axes(skeleton) # se calcula el origen del nuevo sistema de coordenadas
                distance_he_tail = distance(head, tail) # se calcula la distancia entre la cabeza y la cola
                distance_ta_center = distance(tail, center) # se calcula la distancia entre la cola y el origen
                distance_he_center = distance(head, center) # se calcula la distancia entre la cabeza y el origen
                #se añade cada distancia en su array correspondiente
                distance_h_t_array.append(distance_he_tail)
                distance_h_c_array.append(distance_he_center)
                distance_c_t_array.append(distance_ta_center)
                
                ratio = ratio_tail_head(longitud,distance_he_tail) #se calcula el ratio cabeza-cola
                ratio_array.append(ratio) #se añade al array

            #se obtienen las distancias de las líneas que se dibujan a lo largo del eje Y
            distancias = plot_lines_along_line(skeleton) 
            if distancias != "Coil": #si el gusano no esta haciendo coil
                media_distancia = calcular_media(distancias) #se calcula la distancia media
                Avr_dis_array.append(media_distancia) #se añade al array

                distancias_abs = valor_absoluto_array(distancias) #se calcula el valor absoluto de las distancias
                media_distancia_abs = calcular_media(distancias_abs) #se calcula el valor medio
                Avr_abs_dis_array.append(media_distancia_abs) #se añade al array

                max_dist = max(distancias) #se calcula el valor la distancia máxima
                max_dist_array.append(max_dist) #se añade al array

                max_dist_abs = max(distancias_abs) #se calcula el valor la distancia máxima absoluta
                max_abs_dist_array.append(max_dist_abs) #se añade al array

                min_dist = min(distancias) #se calcula el valor la distancia mínima
                min_dist_array.append(min_dist) #se añade al array

                min_dist_abs = min(distancias_abs) #se calcula el valor la distancia mínima absoluta
                min_abs_dist_array.append(min_dist_abs) #se añade al array

                skewness = round(skew(distancias, axis=0, bias=True),3) #se calcula la asimetría de las distancias
                ske_array.append(skewness) #se añade al array

                kurto = round(kurtosis(distancias, axis=0, bias=True),3) #se calcula la curtosis de las distancias
                kurtosis_array.append(kurto) #se añade al array
    
        # Calcular la media de los valores para cada archivo TIFF
        distance_h_t_media = calcular_media_gusano(distance_h_t_array)
        distance_h_c_media = calcular_media_gusano(distance_h_c_array)
        distance_c_t_media = calcular_media_gusano(distance_c_t_array)
        length_media = calcular_media_gusano(length_array)
        ratio_media = calcular_media_gusano(ratio_array)
        Avr_abs_dis_media = calcular_media_gusano(Avr_abs_dis_array)
        Avr_dis_media = calcular_media_gusano(Avr_dis_array)
        max_dist_media = calcular_media_gusano(max_dist_array)
        min_dist_media = calcular_media_gusano(min_dist_array) 
        max_abs_dist_media = calcular_media_gusano(max_abs_dist_array)
        min_abs_dist_media = calcular_media_gusano(min_abs_dist_array)
        ske_media = calcular_media_gusano(ske_array)
        kurtosis_media = calcular_media_gusano(kurtosis_array)

        # Escribir los datos en el archivo CSV
        row_data = [file_name,distance_h_t_media,distance_h_c_media,distance_c_t_media,length_media,ratio_media,Avr_abs_dis_media,Avr_dis_media,
                    max_dist_media,min_dist_media,max_abs_dist_media,min_abs_dist_media,ske_media,kurtosis_media]
        csv_writer.writerow(row_data)

    # Cerrar el archivo CSV
    csv_file.close()
    print("OK")

#función que crea una carpeta en el "output_path" donde se almacenan 3 gráficos de cajas
#en formato PNG correspondientes con los archivos situados en el "folder_path"
def graficos(folder_path, output_path):
    # Inicializar un diccionario para almacenar las listas de datos para cada gráfico
    graficos = {"Ratio head-tail":[],
                "AVG distance":[],
                "AVG distance ABS":[]}
    
    # Inicializar una lista para almacenar los nombres de los archivos TIFF
    nombres_archivos = []
    
    # Obtener una lista de todos los archivos TIFF en el directorio que no sean el esqueleto generado
    tiff_files = [file for file in os.listdir(folder_path) if (file.endswith(".tiff") or file.endswith(".tif")) and not 
                   os.path.splitext(os.path.basename(file))[0].endswith("_skeleton")]
    
    # Iterar sobre cada archivo TIFF en la carpeta
    for file_name in tiff_files:
        # Agregar el nombre del archivo a la lista de nombres de archivos
        nombres_archivos.append(file_name)
        # Inicializar listas para almacenar los datos de ratio y distancia promedio
        ratio_array = []
        Avr_abs_dis_array = []
        Avr_dis_array = []
        # Leer el archivo TIFF
        image_stack = mtif.read_stack(os.path.join(folder_path, file_name))
        # Por cada frame del archivo...
        for i, image in enumerate(image_stack):
            # Procesar la imagen y obtener los datos
            skeleton = obtain_skeleton(image)  # Obtener el esqueleto de la imagen
            longitud = skeleton_length(skeleton)  # Calcular la longitud del esqueleto
            
            points = end_points(skeleton)  # Encontrar los puntos finales del esqueleto
            if len(points) > 1: #si hay mas de un endpoint
                #se obtienen las coordenadas de la cabeza y cola
                head = points[0]
                tail = points[1]
                distance_he_tail = distance(head, tail) #se calcula la distancia entre la cabeza y cola
                ratio = ratio_tail_head(longitud,distance_he_tail) #se calcula el ratio cabeza-cola
                ratio_array.append(ratio) #se añade al array
            
            #se obtienen las distancias de las líneas que se dibujan a lo largo del eje Y
            distancias = plot_lines_along_line(skeleton)
            if distancias != "Coil":  #si el gusano no esta haciendo coil
                media_distancia = calcular_media(distancias) #se calcula la distancia media
                Avr_dis_array.append(media_distancia) #se añade al array

                distancias_abs = valor_absoluto_array(distancias) #se obtiene el array de las distancias en valor absoluto
                media_distancia_abs = calcular_media(distancias_abs)#se calcula el valor medio
                Avr_abs_dis_array.append(media_distancia_abs) #se añade al array
        
        # Agregar las listas de datos a los gráficos correspondientes en el diccionario
        graficos["Ratio head-tail"].append(ratio_array)
        graficos["AVG distance ABS"].append(Avr_abs_dis_array)
        graficos["AVG distance"].append(Avr_dis_array)
        
    # Crear una carpeta para almacenar los gráficos en la ruta de "output_path"
    plots_folder_path = os.path.join(output_path, os.path.basename(folder_path)+"_gráficos")
    os.makedirs(plots_folder_path)
    cmap = plt.get_cmap('tab10')  # Tabla de 10 colores predefinida

    # Iterar sobre cada conjunto de datos en los gráficos
    for clave, valores in graficos.items():
        plt.figure() # Crear una nueva figura
        # Crear un diagrama de caja y bigotes
        bp = plt.boxplot(valores, patch_artist=True, medianprops=dict(color="black")) #mediana en color negro
        # Iterar sobre cada caja y asignar un color del mapa de colores
        for box, color_index in zip(bp['boxes'], range(len(valores))):
            box.set(facecolor=cmap(color_index))  # Asignar el color del mapa de colores a la caja
        
        # Configurar etiquetas en el eje X con los nombres de los archivos
        plt.xticks(ticks=np.arange(1, len(nombres_archivos) + 1), labels=nombres_archivos)
        plt.xlabel('File name')
        plt.ylabel(clave)  # Usar la clave como etiqueta en el eje Y
        # Guardar la figura como un archivo PNG en la carpeta de gráficos
        output_file_path = os.path.join(plots_folder_path, f'{clave}.png')
        plt.savefig(output_file_path)
        # Cerrar la figura para liberar memoria
        plt.close()
    print("OK")
   


#------------------------INTERFAZ GRÁFICA--------------------------------------------------
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Eva\Documents\TFG\proyecto_definitivo\assets\frame0")

# Función para convertir una ruta relativa a la carpeta de activos a una ruta absoluta
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Función para manejar el cierre de la ventana principal de la aplicación
def on_closing():
    window.quit()  # Detiene el bucle principal de Tkinter
    window.destroy()  # Destruye la ventana principal y libera los recursos asociados

# Función para seleccionar una ruta de entrada y actualizar la entrada correspondiente en la interfaz
def select_input_path():
    # Utiliza el cuadro de diálogo de Tkinter para seleccionar una carpeta
    folder_path = tk.filedialog.askdirectory()
    # Borra el contenido actual del campo de entrada de ruta de entrada
    input_entry.delete(0, tk.END)
    # Inserta la ruta seleccionada en el campo de entrada de ruta de entrada
    input_entry.insert(0, folder_path)

# Función para seleccionar una ruta de salida y actualizar la entrada correspondiente en la interfaz
def select_output_path():
    # Utiliza el cuadro de diálogo de Tkinter para seleccionar una carpeta
    folder_path = tk.filedialog.askdirectory()
    # Borra el contenido actual del campo de entrada de ruta de salida
    output_entry.delete(0, tk.END)
    # Inserta la ruta seleccionada en el campo de entrada de ruta de salida
    output_entry.insert(0, folder_path)

#función para redigir al usuario al readme de la app
def know_more_clicked(event):
    instructions = (
        "https://github.com/ParthJadhav/Tkinter-Designer/"
        "blob/master/docs/instructions.md")
    webbrowser.open_new_tab(instructions) #abre la url en el navegador

# Función para seleccionar o deseleccionar una respuesta en la interfaz
def seleccionar_respuesta(idx):
    # Cambia el estado de selección de la respuesta en el índice dado
    respuestas_seleccionadas[idx] = not respuestas_seleccionadas[idx]
    # Actualiza el color de fondo del botón de respuesta correspondiente según su estado de selección
    if respuestas_seleccionadas[idx]: #si la respuesta está seleccionada
        botones_respuesta[idx].config(bg="#F23D3D")  #se aplica el color
    else: #si no está seleccionada
        botones_respuesta[idx].config(bg="#D9D9D9")  #se aplica este color


import threading

# Función para mostrar una barra de progreso mientras se ejecuta el programa
def mostrar_ventana_progreso():
    # Crear una ventana secundaria para mostrar el progreso
    progress_window = tk.Toplevel(window)
    progress_window.title("Progress")
    # Configurar la geometría para centrar la ventana de progreso
    window_width = window.winfo_width()  # Ancho de la ventana principal
    window_height = window.winfo_height()  # Alto de la ventana principal
    progress_window_width = 400  # Ancho de la ventana de progreso
    progress_window_height = 150  # Alto de la ventana de progreso
    # Calcular el desplazamiento para centrar la ventana de progreso
    x_offset = (window_width - progress_window_width) // 2
    y_offset = (window_height - progress_window_height) // 2
    # Establecer la geometría de la ventana de progreso para que esté centrada
    progress_window.geometry(f"400x150+{window.winfo_x() + x_offset}+{window.winfo_y() + y_offset}")
    # Etiqueta de progreso para mostrar un mensaje
    progress_label = tk.Label(progress_window, text="Processing...")
    progress_label.pack(pady=10)  # Empaquetar la etiqueta con un relleno vertical de 10 píxeles
    # Barra de progreso para mostrar el progreso de la tarea
    progress_bar = ttk.Progressbar(progress_window, mode="indeterminate", length=200)  # Modo indeterminado(movimiento continuo), longitud de 200
    progress_bar.pack(pady=10)  # Empaquetar la barra de progreso con un relleno vertical de 10 píxeles
    progress_bar.start()  # Iniciar la animación de la barra de progreso
    # Devolver la ventana de progreso y la barra de progreso
    return progress_window, progress_bar

#función que se encarga de darle lógica a la interfaz llamando a las funciones
def procesamiento(input_path, output_path, elegidas, progress_window, progress_bar):
    # Diccionario que mapea nombres de función a funciones
    diccionario_funciones = {
        "Obtain skeleton files": create_tiff_skeleton,
        "Create CSV": obtener_csv,
        "Create CSV average values": obtener_csv_media,
        "Get graphics": graficos
    }
    # Verificar si hay subcarpetas dentro de la carpeta actual
    subcarpetas = [nombre for nombre in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, nombre))]
    
    # Si no hay subcarpetas, se procesa directamente la carpeta actual
    if not subcarpetas:
        input_path = os.path.join(input_path) #se obtiene la ruta de la carpeta
        seleccionar_funciones(input_path, output_path, elegidas, diccionario_funciones, progress_window)
    else: # Si hay subcarpetas
        # se itera sobre cada una de ellas
        for subcarpeta in subcarpetas:
            input_subpath = os.path.join(input_path, subcarpeta)#se obtiene la ruta de la subcarpeta
            seleccionar_funciones(input_subpath, output_path, elegidas, diccionario_funciones, progress_window)

    progress_bar.stop()  # Detener la barra de progreso
    progress_window.destroy() #destruir la ventana de progreso
    # Mostrar un cuadro de mensaje informando que el proceso ha finalizado con éxito
    tk.messagebox.showinfo("Success!", f"Successfully generated at {output_path}.")

#función que se encarga de llamar a las funciones que haya elegido el usuario
def seleccionar_funciones(input_path, output_path, elegidas, diccionario_funciones, progress_window):
    #se itera sobre las funciones seleccionadas
    for eleccion in elegidas:
        #se obtiene la función correspondiente la opción seleccionada
        funcion = diccionario_funciones.get(eleccion)
        # Verificar si la función existe en el diccionario
        if funcion:
            # Llamar a la función correspondiente
            progress_window.update_idletasks()  # Actualizar la ventana de progreso
            funcion(input_path, output_path) # Llamar a la función correspondiente

def procesar_carpeta2():
    input_path = input_entry.get() # se obtiene la ruta de entrada 
    output_path = output_entry.get() # se obtiene la ruta de salida
    #se obtienen las opciones seleccionadas por el usuario
    elegidas = [respuesta for respuesta, seleccionada in zip(respuestas, respuestas_seleccionadas) if seleccionada]
    # si las rutas de entrada o salida están vacias
    if input_path == "" or output_path == "":
        # Mostrar un mensaje de error si falta información
        tk.messagebox.showinfo("Error!!", "Please, fill in all fields")
    #si no se ha elegido ninguna opción
    elif not elegidas:
        # Mostrar un mensaje de error
        tk.messagebox.showinfo("Error!!", "Please, select at least one option")
    else: #si todos los campos están llenos
        # Mostrar la ventana de progreso en un hilo de fondo
        progress_window, progress_bar = mostrar_ventana_progreso()
        # Crear un hilo para ejecutar el procesamiento de la carpeta
        thread = threading.Thread(target=procesamiento, args=(input_path, output_path, elegidas, progress_window, progress_bar))
        # Iniciar el hilo
        thread.start()

window = Tk()
logo = tk.PhotoImage(file=ASSETS_PATH / "app_icon2.png")
window.call('wm', 'iconphoto', window._w, logo)
window.geometry("862x512")
window.title("RedWorm")
window.configure(bg = "#E9DFEC")
window.protocol("WM_DELETE_WINDOW", on_closing)

#se el fondo de la ventana blanca
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 512,
    width = 862,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
#se crea el rectángulo negro de la izquierda
canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    426.0,
    512.0,
    fill="#242424",
    outline="")

#logo aplicación
image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    213.0,
    109.0,
    image=image_image_2
)
#lína roja en la zona negra
canvas.create_rectangle(
    35.0,
    225.0,
    125.0,
    226.0,
    fill="#F23D3D",
    outline="")
#texto explicativo en la zona izquierda negra
canvas.create_text(
    36.0,
    261.0,
    anchor="nw",
    text="Analyse the behaviour of\n"
    "C.elegans by selecting the\n"
    "desired folder and choose what\n"
    "you want to obtain.",
    fill="#FFFFFF",
    font=("Inter", 22 * -1)
)
know_more = tk.Label(
    text="Click here for instructions",
    bg="#242424", fg="#F23D3D",justify="left", cursor="hand2")
know_more.place(x=36.0, y=399.0)
know_more.bind('<Button-1>', know_more_clicked)


#cuadrado multiplechoice
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    637.0,
    319.0,
    image=image_image_1
)

canvas.create_text(
    490.0,
    238.0,
    anchor="nw",
    text="Select your results:",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)
#boton de run
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=procesar_carpeta2,
    relief="flat"
)
button_1.place(
    x=571.0,
    y=436.0,
    width=133.0,
    height=49.0
)
#input_path
text_box_bg = tk.PhotoImage(file=ASSETS_PATH / "entry_1.png")
input_entry_img = canvas.create_image(637.5, 77.5, image=text_box_bg)
input_entry = tk.Entry(bd=0,   bg="#242424",fg="#FFFFFF",  highlightthickness=0)
input_entry.place(
    x=486.0,
    y=46.0+25,
    width=300.0,
    height=35.0
)
input_entry.focus()

path_picker_img = tk.PhotoImage(file = ASSETS_PATH / "folder.png")
path_picker_button = tk.Button(
    image = path_picker_img,
    text = '',
    compound = 'center',
    fg = 'white',
    borderwidth = 0,
    highlightthickness = 0,
    command=select_input_path,
    relief = 'flat')

path_picker_button.place(
    x=745.0,
    y=58.0,
    width=37.0,
    height=38.0
)
canvas.create_text(
    484.0,
    53.0,
    anchor="nw",
    text="Input Path",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)
#outputpath
output_entry_img = canvas.create_image( 637.5, 171.5, image=text_box_bg)
output_entry = tk.Entry(bd=0,bg="#242424",fg="#FFFFFF",  highlightthickness=0)
output_entry.place(
    x=486.0,
    y=140.0+25,
    width=300.0,
    height=35.0
)
output_entry.focus()

pathOut_picker_img = tk.PhotoImage(file = ASSETS_PATH / "folder.png")
pathOut_picker_button = tk.Button(
    image = pathOut_picker_img,
    text = '',
    compound = 'center',
    fg = 'white',
    borderwidth = 0,
    highlightthickness = 0,
    command=select_output_path,
    relief = 'flat')
pathOut_picker_button.place(
    x=745.0,
    y=152.0,
    width=37.0,
    height=38.0
)

canvas.create_text(
    484.0,
    147.0,
    anchor="nw",
    text="Output Path",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

# Lista de opciones disponibles para el usuario
respuestas = ["Obtain skeleton files", "Create CSV", "Create CSV average values", "Get graphics"] 
# Lista de valores booleanos que indican si cada opción ha sido seleccionada o no
respuestas_seleccionadas = [False] * len(respuestas)
# Lista que contendrá los botones creados para cada opción
botones_respuesta = []
# Posición vertical inicial donde se colocará el primer botón
pos_y = 267
# Creación de botones para cada opción
for i, respuesta in enumerate(respuestas):
    # Crear un botón con el texto de la opción actual
    boton = tk.Button(text=respuesta, width=20, command=lambda idx=i: seleccionar_respuesta(idx))
    boton.place(x=570.0, y=pos_y) #Colocar el botón en la ventana gráfica
    botones_respuesta.append(boton) #Agregar el botón a la lista de botones
    pos_y += 35  # Incrementar la posición vertical para el siguiente botón

window.resizable(False, False)
window.mainloop()