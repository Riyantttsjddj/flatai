from flask import Flask, render_template, request, Response, redirect, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['DOWNLOAD_TOKENS'] = {}

# Constants
OUTPUT_VIDEO = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
OUTPUT_IMAGE = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

# Load TFLite model
MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect(image_np, threshold=0.5):
    input_data = np.expand_dims(image_np, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = []
    try:
        x_center, y_center, w, h, conf = output
        for i in range(conf.shape[0]):
            if conf[i] >= threshold:
                xc, yc, bw, bh = x_center[i], y_center[i], w[i], h[i]
                xmin = max(int((xc - bw / 2) * width), 0)
                ymin = max(int((yc - bh / 2) * height), 0)
                xmax = min(int((xc + bw / 2) * width), width)
                ymax = min(int((yc + bh / 2) * height), height)
                boxes.append((xmin, ymin, xmax, ymax))
    except:
        pass
    return boxes

def process_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (width, height))
    input_image = image_resized.astype(np.float32) / 255.0
    boxes = detect(input_image)
    
    # Draw bounding boxes and count
    count = len(boxes)
    cv2.putText(image_resized, f"Objects: {count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(OUTPUT_IMAGE, image_resized)
    return OUTPUT_IMAGE, count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file selected", 400
        
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
        
    if not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Determine if file is image or video
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    if file_ext in {'jpg', 'jpeg', 'png'}:
        output_path, count = process_image(filepath)
        os.remove(filepath)  # Clean up original
        return render_template('result.html', 
                             media_type='image',
                             filename=os.path.basename(output_path),
                             count=count)
    else:
        # Video processing
        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

        total_count = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_resized = cv2.resize(frame, (width, height))
            input_frame = frame_resized.astype(np.float32) / 255.0
            boxes = detect(input_frame)
            
            count = len(boxes)
            total_count += count
            frame_count += 1
            
            cv2.putText(frame_resized, f"Objects: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            out.write(frame_resized)

        cap.release()
        out.release()
        os.remove(filepath)  # Clean up original
        
        avg_count = total_count // max(1, frame_count)
        return render_template('result.html',
                             media_type='video',
                             filename='output.mp4',
                             count=avg_count)

@app.route('/download/<filename>')
def download_file(filename):
    if filename not in ['output.mp4', 'output.jpg']:
        return "Invalid file", 404
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(filepath)
    
    if range_header:
        # Handle partial content
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start)
        end = int(end) if end else file_size - 1
        
        def generate_chunk():
            with open(filepath, 'rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        response = Response(
            generate_chunk(),
            206,
            mimetype='video/mp4' if filename.endswith('.mp4') else 'image/jpeg',
            content_type='video/mp4' if filename.endswith('.mp4') else 'image/jpeg',
            direct_passthrough=True
        )
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(end - start + 1))
    else:
        # Full download
        response = send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=True,
            mimetype='video/mp4' if filename.endswith('.mp4') else 'image/jpeg'
        )
        response.headers['Content-Length'] = file_size
    
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000)