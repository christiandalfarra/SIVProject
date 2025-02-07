import cv2
import numpy as np

# Load or create a blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Example: Generate random contours (Replace this with actual contours)
contours = [np.random.randint(100, 400, (5, 1, 2), dtype=np.int32) for _ in range(5)]

# Compute convex hulls for each contour
hulls = [cv2.convexHull(cnt) for cnt in contours]

# Find the largest convex hull by area
largest_hull = max(hulls, key=cv2.contourArea)

# Draw all hulls in blue
for hull in hulls:
    cv2.drawContours(image, [hull], -1, (255, 0, 0), 2)
cv2.imshow("All contours",image)

# Draw the largest hull in green
cv2.drawContours(image, [largest_hull], -1, (0, 255, 0), 3)

# Print and display the area of the largest hull
largest_area = cv2.contourArea(largest_hull)
print("Largest Convex Hull Area:", largest_area)

# Show the image
cv2.imshow("Largest Convex Hull", image)
cv2.waitKey(0)
cv2.destroyAllWindows()