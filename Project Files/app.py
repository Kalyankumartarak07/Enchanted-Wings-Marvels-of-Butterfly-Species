from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import uuid
import PIL
app = Flask(__name__)
model = load_model("butterfly_mobilenetv2_model.h5")
class_labels = ['AMERICAN SNOOT', 'SCARCE SWALLOW', 'WHITE LINED SPHINX MOTH', 'ZEBRA LONG WING']
STATIC_DIR = "/Users/kalyankumartarak/Desktop/butterfly-classification/butterfly_classifier_app/static"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in request."

        file = request.files['file']
        if file.filename == '':
            return "No file selected."

        if not file.mimetype.startswith("image"):
            return "❌ Only image files are allowed. Please upload JPG or PNG."
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(STATIC_DIR, unique_filename)
        file.save(filepath)
        try:
            img = image.load_img(filepath, target_size=(224, 224))
        except PIL.UnidentifiedImageError:
            return "⚠️ The uploaded file could not be identified as an image."

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))

        if class_index >= len(class_labels):
            return f"⚠️ Predicted class index {class_index} is out of range."

        result = class_labels[class_index]

        return render_template("result.html", prediction=result, image_path=unique_filename)

    return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)
