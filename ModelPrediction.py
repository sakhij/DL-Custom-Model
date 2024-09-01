#prediction
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def prepare_image(img_path, target_size):
    # Load image
    img = image.load_img(img_path, target_size=target_size)

    # Convert to array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data if required
    img_array /= 255.0  # Assuming the model expects pixel values in [0, 1]

    return img_array

def predict_image(model, img_array):
    # Make prediction
    predictions = model.predict(img_array)

    return predictions

def interpret_prediction(predictions, class_labels):
    # Get the class index with the highest probability
    predicted_index = np.argmax(predictions, axis=1)[0]
    # Get the class name
    predicted_class = class_labels[predicted_index]
    return predicted_class

# Example usage
def run_prediction(img_path, model_path):
    # Load the model
    model = load_model(model_path)

    # Prepare the image
    target_size = (32, 32)  # Example target size, adjust based on your model
    img_array = prepare_image(img_path, target_size)

    # Predict the image
    predictions = predict_image(model, img_array)

    # Interpret the prediction
    predicted_class = interpret_prediction(predictions,class_labels)

    print(f'Predicted class: {predicted_class}')

# Entry point for predicting an image
img_path = 'download.jpeg'
model_path = 'cifar10_model.h5'
run_prediction(img_path, model_path)
