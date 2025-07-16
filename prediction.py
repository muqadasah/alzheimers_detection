import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to your MRI image
image_path = "D:\Desktop\Screenshot 2024-07-01 212910.png" 
# image_path = r"D:\Desktop\26 (19).jpg" #mild demented
# image_path = r"D:\Desktop\27 (2).jpg"   
# image_path = r"D:\Desktop\26 (62).jpg"
# image_path = r"D:\Desktop\29 (70).jpg"

# Load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(176, 208)):  # Adjust target size to match model's expected input size
    img = image.load_img(image_path, target_size=target_size)  # Resize image to the model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Load the pre-trained model
model = tf.keras.models.load_model("alzheimer's_detection.h5")  # Adjust the path to your saved model

# Load and preprocess the image
img_array = load_and_preprocess_image(image_path)

# Run the inference
predictions = model.predict(img_array)
print('Predictions:', predictions)

# Assuming the model outputs probabilities for each class, interpret the result
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']  # Adjust these class names according to your model's output
predicted_class = class_names[np.argmax(predictions)]
print('Predicted class:', predicted_class)

# Optional: Display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.title(f'Predicted: {predicted_class}')
plt.show()