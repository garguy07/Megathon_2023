import cv2
import numpy as np

# Load the input image (assumes it is in BGR format)
input_image = cv2.imread('paddy.jpg')

# Convert the image to float32 for calculations
input_image = input_image.astype(np.float32)

# Find the darkest pixel values in the image
dark_object = np.min(input_image, axis=2)

# Estimate the atmospheric correction value (the dark object value)
atmospheric_correction = np.percentile(dark_object, 5)  # You can adjust the percentile if needed

# Apply the correction to each color channel
corrected_image = input_image - atmospheric_correction

# Clip the reflectance values to [0, 255]
corrected_image = np.clip(corrected_image, 0, 255)

# Convert the corrected image back to uint8 for display or further processing
corrected_image = corrected_image.astype(np.uint8)

# Save or display the corrected image
cv2.imwrite('atmospherically_corrected_image.jpg', corrected_image)

# Optionally, display the original and corrected images
cv2.imshow('Original Image', input_image.astype(np.uint8))
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
