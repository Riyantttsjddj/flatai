
from flask import Flask, render_template, request, send_file
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
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output[0]
    x_center, y_center, w, h, conf = output

    boxes = []
    for i in range(conf.shape[0]):
        if conf[i] >= threshold:
            xc, yc, bw, bh = x_center[i], y_center[i], w[i], h[i]
            xmin = max(int((xc - bw / 2) * width), 0)
            ymin = max(int((yc - bh / 2) * height), 0)
            xmax = min(int((xc + bw / 2) * width), width)
            ymax = min(int((yc + bh / 2) * height), height)
            boxes.append((xmin, ymin, xmax, ymax))
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
    return send_file(OUTPUT_VIDEO, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
