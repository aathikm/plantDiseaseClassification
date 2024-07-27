from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from prediction_pipeline import predict_image  # Ensure this imports your prediction function

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

@app.route('/')
def index():
    return "Image Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
            file.save(file_path)

        try:
            # Call your prediction function with the temporary file path
            result = predict_image(file_path)
        finally:
            # Remove the temporary file
            os.remove(file_path)
        
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
