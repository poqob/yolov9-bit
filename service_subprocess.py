import os
import time
import subprocess
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from pymongo import MongoClient
import base64

CONF_THRES = 0.25

app = Flask(__name__, static_folder='static', static_url_path='/')

# MongoDB connection (adjust URI as needed)
client = MongoClient('mongodb://localhost:27017/')
db = client['ekg_vision']
scan_results = db['scan_results']

@app.route('/detect', methods=['POST'])
def detect():
    # Allow model selection
    model_name = request.form.get('model_name', 'implementation-residual_h_swish_SGD')
    # Handle user_id and patient_id fields
    user_id = request.form.get('user_id')
    patient_id = request.form.get('patient_id')
   
    # Sanitize model name (only allow alphanumeric, dash, underscore)
    import re
    if not re.match(r'^[\w\-]+$', model_name):
        return jsonify({'error': 'Invalid model name'}), 400
    weights = f"runs/train/{model_name}/weights/best.pt"
    if not os.path.exists(weights):
        return jsonify({'error': f'Model weights not found: {weights}'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)

    # Read image as base64 for MongoDB storage
    image_file.seek(0)
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_file.seek(0)  # Reset for saving to disk

    # Create a temp directory for this request
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, filename)
        image_file.save(img_path)

        # Output directory for detect_dual.py
        outdir = os.path.join(tmpdir, 'out')
        os.makedirs(outdir, exist_ok=True)

        # Run detect_dual.py as subprocess
        cmd = [
            'python3', 'detect_dual.py',
            '--weights', weights,
            '--source', img_path,
            '--conf-thres', str(CONF_THRES),
            '--save-txt', '--save-conf',
            '--project', outdir,
            '--name', 'exp',
            '--exist-ok'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Detection failed', 'stderr': result.stderr}), 500

        # Find the label file
        exp_dir = os.path.join(outdir, 'exp')
        label_dir = os.path.join(exp_dir, 'labels')
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not label_files:
            return jsonify({'boxes': [], 'image_url': None})
        label_path = os.path.join(label_dir, label_files[0])

        # Parse YOLO label file
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:
                    cls, x, y, w, h, conf = parts
                    boxes.append({
                        'class': int(float(cls)),
                        'x_center': float(x),
                        'y_center': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(conf)
                    })

        # Optionally, copy the annotated image to a tmp file and return its URL
        result_img_path = os.path.join(exp_dir, filename)
        image_url = None
        result_image_b64 = None
        if os.path.exists(result_img_path):
            import uuid
            tmp_img_name = f"tmp_{uuid.uuid4().hex}.jpg"
            tmp_img_dir = os.path.join('static', 'tmp')
            os.makedirs(tmp_img_dir, exist_ok=True)
            tmp_img_path = os.path.join(tmp_img_dir, tmp_img_name)
            with open(result_img_path, 'rb') as src, open(tmp_img_path, 'wb') as dst:
                img_bytes = src.read()
                dst.write(img_bytes)
                result_image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            image_url = f"/tmp/{tmp_img_name}"

        # Save to MongoDB only if all required fields are present
        try:
            scan_results.insert_one({
                'user_id': user_id,
                'patient_id': patient_id,
                'model_name': model_name,
                'date': datetime.now(timezone.utc),
                'image': image_b64,
                'result_image': result_image_b64,
                'boxes': boxes
            })
        except Exception as e:
            print(f"MongoDB insert error: {e}")

        return jsonify({'boxes': boxes, 'image_url': image_url})

@app.route('/tmp/<filename>')
def serve_tmp_image(filename):
    return send_from_directory(os.path.join('static', 'tmp'), filename)

@app.route('/models', methods=['GET'])
def list_models():
    models_dir = os.path.join('runs', 'train')
    try:
        models = []
        for name in os.listdir(models_dir):
            weights_path = os.path.join(models_dir, name, 'weights', 'best.pt')
            if os.path.isfile(weights_path):
                models.append(name)
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
