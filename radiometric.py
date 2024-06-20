import cv2
import numpy as np

# Load the input image (grayscale)
input_image = cv2.imread('paddy.jpg', cv2.IMREAD_GRAYSCALE)

# Define sensor-specific calibration parameters
dark_pixel_value = 1  # Replace with the dark pixel value from your calibration
bright_pixel_value = 200  # Replace with the bright pixel value from your calibration
calibration_factor = 1.0 / (bright_pixel_value - dark_pixel_value)  # Calculate the calibration factor

# Apply radiometric calibration
radiometrically_calibrated = (input_image - dark_pixel_value) * calibration_factor

# Clip the result to [0, 255] and convert to 8-bit
calibrated_image = np.clip(radiometrically_calibrated, 0, 255).astype(np.uint8)

# Save the calibrated image
cv2.imwrite('radiometrically_calibrated_image.jpg', calibrated_image)
