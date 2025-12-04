import os
import cv2
import numpy as np
import random  # <--- Added this to shuffle files
from flask import Flask, render_template, request, url_for
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'train')

IMG_SIZE = 64
LIMIT = 2000

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

data = []
labels = []

if os.path.exists(DATA_DIR):
    files = os.listdir(DATA_DIR)
    
    # --- THE FIX: Shuffle to get both Cats and Dogs ---
    random.shuffle(files)
    files = files[:LIMIT]
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img.flatten())
            if 'cat' in f:
                labels.append(0)
            else:
                labels.append(1)

X = np.array(data)
y = np.array(labels)

# Check if we actually have 2 classes now
if len(np.unique(y)) < 2:
    print("Error: Still found only 1 class. Make sure 'data/train' has BOTH cat and dog images.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                vec = img.flatten().reshape(1, -1)
                pred = model.predict(vec)[0]
                prediction = "Dog" if pred == 1 else "Cat"
                img_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5002)