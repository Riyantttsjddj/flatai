from flask import Flask, render_template, request, Response, redirect, flash, send_file
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'rahasia'
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['DOWNLOAD_TOKENS'] = {}

MODEL_PATH = 'model.tflite'

# ... (kode sebelumnya tetap sama) ...

@app.route('/download')
def download_page():
    # Halaman untuk memulai download
    if not os.path.exists(OUTPUT_VIDEO):
        flash('Video hasil belum tersedia. Harap proses video terlebih dahulu.')
        return redirect('/')
    
    # Generate token unik untuk download
    download_token = str(datetime.now().timestamp())
    app.config['DOWNLOAD_TOKENS'][download_token] = OUTPUT_VIDEO
    
    return render_template('download.html', token=download_token)

@app.route('/download_video/<token>')
def download_video(token):
    if token not in app.config['DOWNLOAD_TOKENS']:
        return "Token download tidak valid", 404
    
    video_path = app.config['DOWNLOAD_TOKENS'][token]
    
    # Hapus token setelah digunakan untuk mencegah download berulang
    app.config['DOWNLOAD_TOKENS'].pop(token, None)
    
    # Mendukung pause/resume download
    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(video_path)
    
    if range_header:
        # Parsing range header
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start)
        end = int(end) if end else file_size - 1
        
        # Membaca bagian file yang diminta
        with open(video_path, 'rb') as f:
            f.seek(start)
            data = f.read(end - start + 1)
        
        # Membuat response partial content
        response = Response(
            data,
            206,
            mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(end - start + 1))
    else:
        # Download biasa jika tidak ada range header
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='hasil_deteksi.mp4'
        )
        response.headers['Content-Length'] = file_size
    
    response.headers['Cache-Control'] = 'no-store'
    return response

# ... (kode lainnya tetap sama) ...
