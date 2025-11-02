# app.py

from flask import Flask, render_template, request
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2

# -----------------------------
# 1. Flask setup
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# -----------------------------
# 2. Load trained model
# -----------------------------
model = tf.keras.models.load_model("FACE_DETECTION/face_emotionModel.h5")



# Emotion labels (must match training order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# 3. Database setup
# -----------------------------
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  image_path TEXT,
                  emotion TEXT,
                  date_uploaded TEXT)''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# 4. Predict emotion function
# -----------------------------
def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]

# -----------------------------
# 5. Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion = predict_emotion(filepath)

        # Generate friendly message
        if emotion == 'Sad':
            message = "You are frowning. Why are you sad?"
        elif emotion == 'Happy':
            message = "You look happy! Keep smiling!"
        elif emotion == 'Angry':
            message = "You seem angry. Take a deep breath!"
        elif emotion == 'Surprise':
            message = "You look surprised! What happened?"
        elif emotion == 'Fear':
            message = "You look scared. Everything okay?"
        elif emotion == 'Disgust':
            message = "You look disgusted. Did something bother you?"
        else:
            message = "You look calm and neutral."

        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, image_path, emotion, date_uploaded) VALUES (?, ?, ?, ?, ?)",
                  (name, email, filepath, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return f"""
        <h1>Emotion Detection Result</h1>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Detected Emotion:</strong> {emotion}</p>
        <p>{message}</p>
        <img src='/{filepath}' width='200'>
        <br><br>
        <a href='/'>Go back</a>
        """
    else:
        return "No image uploaded!"

# -----------------------------
# 6. Run the app
# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

