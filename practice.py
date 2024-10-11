import cv2

image = cv2.imread('image.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

resized_image = cv2.resize(grayscale_image, None, fx=0.5, fy =0.5)

cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Resized Image', resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
