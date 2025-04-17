import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2  # OpenCV to read and process images

def create_model(input_shape, num_classes):
    """
    Create a simple CNN model for classifying waste images.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def load_sample_data(image_path):
    """
    Load and preprocess the image for prediction.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Image not found at path: " + image_path)
    # Resize the image to 128x128 pixels
    image = cv2.resize(image, (128, 128))
    # Normalize the image to range [0, 1]
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def predict_waste_category(model, image_path):
    """
    Predict the waste category of a sample image.
    """
    input_image = load_sample_data(image_path)
    predictions = model.predict(input_image)
    category_idx = np.argmax(predictions)
    # Define the waste categories
    categories = ['plastic', 'metal', 'paper', 'glass']
    return categories[category_idx]

if __name__ == "__main__":
    # Define CNN model parameters
    input_shape = (128, 128, 3)
    num_classes = 4  # plastic, metal, paper, glass

    # Create the model
    model = create_model(input_shape, num_classes)
    
    # In a real project, you would train your model on labeled data
    # For this example, we assume the model is already trained.
    # To simulate a trained model without training, you could load weights:
    # model.load_weights('model_weights.h5')
    
    # Compile the model for demonstration purposes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # For demonstration, we are performing a prediction on a sample image.
    sample_image_path = 'sample_waste.jpg'  # Ensure this image exists in your project folder.
    try:
        category = predict_waste_category(model, sample_image_path)
        print("Predicted waste category:", category)
    except Exception as e:
        print("Error during prediction. Please ensure 'sample_waste.jpg' is available. Error:", e)
