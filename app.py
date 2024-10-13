from flask import Flask, request, after_this_request, jsonify
from werkzeug.utils import secure_filename
from audio_model_converter import audio_model_converter
import os
import traceback

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=['GET'])
def hi():
    return jsonify({"hi abgd": 1}), 200

@app.route("/analyze/", methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(file_path)
        file.save(file_path)

        @after_this_request
        def remove_file(response):
            try:
                os.remove(file_path)
                print(f"File {file_path} removed successfully")
            except Exception as e:
                print(f"Error removing file: {e}")
            return response

        try:
            result = audio_model_converter(filename)
            return jsonify({"label": result}), 200
        except Exception as e:
            traceback.print_exc()  # Print the traceback for debugging
            return jsonify({"error": f"{str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a '.wav' file."}), 400

if __name__ == '__main__':
    # Use dynamic host and port
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=1)
