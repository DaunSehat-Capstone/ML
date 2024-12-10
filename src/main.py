import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model('assets/model/model.h5')
# model.summary()

def preprocess_image(image_path, target_size=(120, 120)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict_image(image_path, class_names):
    image = preprocess_image(image_path)
    predictions = model.predict(image, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return class_names[predicted_class[0]], confidence

def print_result(predicted_class, confidence):
    result = {}
    if confidence < 0.9:
        result['plant'] = 'Tanaman tidak diketahui'
    else:
        with open('assets/data/plant.csv', mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                if predicted_class in lines:
                    result['plant'] = lines[1]
                    result['condition'] = lines[2]
                    result['treatment'] = lines[3]
                    result['confidence'] = str(confidence)
                    break
    return result

class_names = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy",
    "Blueberry___healthy", 
    "Cherry___healthy", 
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", 
    "Corn___Common_rust", 
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight", 
    "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy", 
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", 
    "Peach___Bacterial_spot", 
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", 
    "Pepper,_bell___healthy", 
    "Potato___Early_blight",
    "Potato___healthy", 
    "Potato___Late_blight", 
    "Raspberry___healthy", 
    "Soybean___healthy",
    "Squash___Powdery_mildew", 
    "Strawberry___healthy", 
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", 
    "Tomato___Early_blight", 
    "Tomato___healthy",
    "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", 
    "Tomato___Tomato_mosaic_virus", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# img_path = "assets/img/padi.jpeg"
img_path = "assets/img/corn_common_rust.jpg"
# img_path = "assets/img/strawberry_lead_scorch.jpg"
predicted_class, confidence = predict_image(img_path, class_names)
result = print_result(predicted_class, confidence)
print(result)


