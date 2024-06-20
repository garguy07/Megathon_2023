import cv2
import numpy as np
import csv

def calculate_ndvi(image_path):
    # Load the satellite image
    image = cv2.imread(image_path)

    # Convert the image to floating-point format
    image = image.astype(float)

    # Split the image into Red and NIR channels
    red_band = image[:, :, 2]
    nir_band = image[:, :, 1]

    # Calculate NDVI, handling division by zero
    denominator = nir_band + red_band
    ndvi = np.where(denominator != 0, (nir_band - red_band) / denominator, 0)  # Replace '0' with a suitable value

    # Optionally, scale NDVI to the range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi

if __name__ == '__main__':
    image_path = '/home/yeshu/Desktop/megathon/sow_2.jpeg'  # Replace with your image file path
    ndvi_map = calculate_ndvi(image_path)

    # Print the NDVI value for the entire image
    average_ndvi = np.mean(ndvi_map)
    print(f"Average NDVI Value for sowing plant: {average_ndvi:.4f}")

    output_csv_file = 'ndvi_values.csv'
    with open(output_csv_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["harvest", f"{average_ndvi:.4f}"])


    # with open(output_csv_file, 'r') as file:
    #         csv_reader = csv.reader(file)
            
    #         # Read and print a specific column (e.g., the second column)
    #         for row in csv_reader:
    #             if len(row) >= 2:  # Check if the row has at least two columns
    #                 print(row[1])