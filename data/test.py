import cv2
import numpy as np

image = cv2.imread('cosmetic.jpg')
image_gray = cv2.imread('cosmetic.jpg', cv2.IMREAD_GRAYSCALE)

b, g, r = cv2.split(image)
image2 = cv2.merge([r, g, b])

cv2.imshow('Image', image2)
cv2.waitKey(0)

cv2.imshow('Image_gray', image_gray)
cv2.waitKey(0)

blur = cv2.GaussianBlur(image_gray, ksize=(3, 3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edged = cv2.Canny(blur, 10, 250)
cv2.imshow('Edged', edged)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closed', closed)
cv2.waitKey(0)

contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
contours_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours_image', contours_image)
cv2.waitKey(0)

contours_xy = np.array(contours)

# Find x_min, x_max, y_min, y_max
x_min, x_max, y_min, y_max = np.inf, -1, np.inf, -1

for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        x = contours_xy[i][j][0][0]
        y = contours_xy[i][j][0][1]
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

# Crop and save the trimmed image
img_trim = image[y_min:y_max, x_min:x_max]
cv2.imwrite('org_trim.jpg', img_trim)

# Read and display the trimmed image
org_image = cv2.imread('org_trim.jpg')
cv2.imshow('Trimmed Image', org_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
