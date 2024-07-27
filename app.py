from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from src.pipeline.prediction_pipeline import PredictionPipeline  # Make sure this imports your prediction function

from src.loggingInfo.loggingInfo import logging

app = Flask(__name__)
# CORS(app)
CORS(app, resources={f"/*": {"origins": "*"}}) 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.info("the predction function triggered")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        logging.info("file is null")
        return redirect(url_for('index'))
    
    if file:
        logging.info("file existing")
        filename = secure_filename(file.filename)
        logging.info(f"fileName: {filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"fileName: {file_path}")
        file.save(file_path)
        logging.info(f"file saved")
        
        # Call your prediction function here
        predition_obj = PredictionPipeline(file_path)
        result = predition_obj.predict()
        logging.info(f"fileName: {result}")
        
        return jsonify(result_val=result) #render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
