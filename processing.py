import cv2

# Load the image
image = cv2.imread('paddy.jpg')

# # Convert the image to grayscale (optional)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply contrast adjustment using CLAHE (Contrast Limited Adaptive Histogram Equalization)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_image = clahe.apply(gray_image)

# Apply noise reduction using Gaussian blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Perform color correction (optional)
# You can use color correction techniques specific to your image data

# Save the preprocessed image
cv2.imwrite('preprocessed_image.jpg', blurred_image)

# Display the original and preprocessed images (optional)
cv2.imshow('Original Image', image)
cv2.imshow('Preprocessed Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
