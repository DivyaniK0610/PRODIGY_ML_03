import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

# Create uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(64, 64))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = i / 255.0
    p = model.predict(i)
    if p[0][0] > 0.5:
        return "Dog"
    else:
        return "Cat"

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        # Save the image
        img_path = os.path.join("uploads", img.filename)
        img.save(img_path)
        
        # Get Prediction
        p = predict_label(img_path)
        
        # CRITICAL: Pass the 'img_name' back to the HTML
        return render_template("index.html", prediction=p, img_name=img.filename)
    return render_template("index.html")

# --- NEW ROUTE: This allows the HTML to load images from the 'uploads' folder ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ =='__main__':
    app.run(debug=True)