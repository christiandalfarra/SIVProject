import cv2
import numpy as np

def identify_road_color(image, roi_coords):
    """Identifica il colore medio della strada basandosi su una regione di interesse (ROI)."""
    roi = image[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    mean_color = np.mean(roi, axis=(0, 1))
    std_color = np.std(roi, axis=(0, 1))
    lower_bound = np.clip(mean_color - 3 * std_color, 0, 255)
    upper_bound = np.clip(mean_color + 3 * std_color, 0, 255)
    return lower_bound.astype(np.uint8), upper_bound.astype(np.uint8)

def extract_road_contours(image, lower_bound, upper_bound):
    """Estrae i contorni della strada basandosi sul modello di colore."""
    mask = cv2.inRange(image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def convex_hull_contours(contours):
    """Applica il convex hull per raffinare i contorni."""
    return [cv2.convexHull(cnt) for cnt in contours]

def process_image(image):
    """Elabora l'immagine per evidenziare i bordi della strada."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # L'immagine è già in scala di grigi
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def refine_edges(edges):
    """Raffina i bordi applicando operazioni morfologiche."""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    return dilated

def detect_potholes(edges, min_size=50):
    """Identifica le buche basandosi sui contorni scuri all'interno della strada."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potholes = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size]
    return potholes

def main(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (1000, 750))
    roi_coords = (200, 200, 800, 500)  # Esempio di ROI sopra il cofano (x_min,y_min,x_max,y_max)
    lower_bound, upper_bound = identify_road_color(image_resized, roi_coords)
    _, mask = extract_road_contours(image, lower_bound, upper_bound)
    refined_edges = refine_edges(process_image(mask))
    potholes = detect_potholes(refined_edges)
    
    output = image.copy()
    cv2.drawContours(output, potholes, -1, (0, 0, 255), 2)
    output_resized = cv2.resize(output, (1000, 750))  # Ridimensiona l'output
    cv2.imshow("Detected Potholes", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test_images/test3.jpg")