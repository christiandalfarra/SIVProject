import cv2
import numpy as np

def detect_potholes(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Immagine non trovata o percorso errato!")
    
    # Converti in scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica un filtro gaussiano per ridurre il rumore
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Applica il filtro Canny per il rilevamento dei bordi
    edges = cv2.Canny(blurred_image, 50, 150)

    # Dilatazione per eliminare bordi non desiderati e unire quelli vicini
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Trova i contorni
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni in base alla dimensione (modello delle buche)
    potholes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 2500:  # Dimensioni approssimative delle buche
            potholes.append(contour)

    # Disegna i contorni rilevati sull'immagine originale
    for contour in potholes:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    # Mostra il risultato
    cv2.imshow('Pothole Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Percorso dell'immagine da analizzare
image_path = 'image1.png'
detect_potholes(image_path)
