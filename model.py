import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from skin import model_cnn, label_encoder  # Import your model and label encoder

app = Flask(__name__)

# Define a function to process the uploaded image and make a prediction
def predict_skin_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model_cnn.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    class_name = label_encoder.inverse_transform(predicted_class)[0]
    return class_name

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploaded_images', filename)
        file.save(file_path)
        predicted_class = predict_skin_disease(file_path)
        return f'Predicted Skin Disease: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)