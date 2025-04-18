from flask import Flask, render_template, request, Response, redirect
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
MODEL_PATH = 'model.tflite'
UPLOAD_FOLDER = 'static'
OUTPUT_VIDEO = os.path.join(UPLOAD_FOLDER, 'output.mp4')

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if not file:
        return "No file uploaded", 400
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (width, height)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        input_frame = frame_resized.astype(np.float32) / 255.0
        boxes = detect(input_frame)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame_resized)

    cap.release()
    out.release()
    return redirect('/download')

@app.route('/download')
def download_file():
    # Cek jika file output ada
    if not os.path.exists(OUTPUT_VIDEO):
        return "Output video not found", 404
    
    file_size = os.path.getsize(OUTPUT_VIDEO)
    range_header = request.headers.get('Range', None)

    if range_header:
        # Handle pause/resume download
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start)
        end = int(end) if end else file_size - 1
        
        def generate_chunk():
            with open(OUTPUT_VIDEO, 'rb') as f:
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
            206,  # Partial content status code
            mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(end - start + 1))
    else:
        # Normal download
        def generate():
            with open(OUTPUT_VIDEO, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        response = Response(
            generate(),
            mimetype='video/mp4'
        )
        response.headers['Content-Length'] = file_size

    # Common headers
    response.headers['Content-Disposition'] = 'attachment; filename=output.mp4'
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)