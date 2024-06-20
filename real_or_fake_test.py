import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = keras.models.load_model('real_or_fake_model.h5')

# Load and preprocess the new image
img_path = 'rice001ahf.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make a prediction
predictions = model.predict(img_array)

# Get the class label with the highest probability
class_index = np.argmax(predictions)
class_labels = ['fake', 'real']
predicted_crop = class_labels[class_index]

# Display the prediction
print(f"The image is predicted to be: {predicted_crop}")