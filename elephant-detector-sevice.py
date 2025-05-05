import os
import io
import datetime
import json
import csv
import time
import cv2
import threading

from flask import (
    Flask,
    request,
    render_template_string,
    redirect,
    url_for,
    Response,
    jsonify
)
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Base directory
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

# Static subfolders
UPLOAD_SUBFOLDER = 'uploads'
PROCESSED_SUBFOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static', UPLOAD_SUBFOLDER)
app.config['PROCESSED_FOLDER'] = os.path.join(basedir, 'static', PROCESSED_SUBFOLDER)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'reports.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB and model
db = SQLAlchemy(app)
model = YOLO(os.path.join(basedir, 'yolo11n_1.pt'))

# Progress tracking
PROGRESS = {}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Supported image extensions
IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'bmp'}

# Database model
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    media_type = db.Column(db.String(10))
    original_path = db.Column(db.String(300))
    processed_path = db.Column(db.String(300))
    details = db.Column(db.Text)

with app.app_context():
    db.create_all()

# Shared CSS for dark theme with orange accent
shared_css = '''
<style>
  body { background-color: #121212; color: #FFFFFF; }
  a, .btn-link { color: #FF5722 !important; }
  .btn-primary, .btn-secondary, .btn-success {
    background-color: #FF5722 !important;
    border-color: #FF5722 !important;
    color: #FFFFFF !important;
  }
  .btn-primary:hover, .btn-secondary:hover, .btn-success:hover {
    background-color: #E64A19 !important;
    border-color: #E64A19 !important;
    color: #FFFFFF !important;
  }
  table.table { background-color: #1E1E1E; color: #FFFFFF; }
  table.table th, table.table td {
    color: #FFFFFF !important;
    border-color: #333333 !important;
  }
  /* Progress bar styling */
  #uploadProgress::-webkit-progress-value { background-color: #FF5722; }
  #uploadProgress::-moz-progress-bar    { background-color: #FF5722; }
  #uploadProgress { background-color: #333; }
  /* Table striping */
  .table-striped tbody tr:nth-of-type(odd)  { background-color: #1E1E1E; }
  .table-striped tbody tr:nth-of-type(even) { background-color: #242424; }
  /* Video container */
  .video-container {
    position: relative;
    width: 100%;
    padding-top: 56.25%;
    margin-bottom: 1rem;
  }
  .video-container video {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: contain;
  }
  #progress-container { margin-top: 1rem; display: none; }
  #uploadProgress { width: 100%; height: 1.5rem; background: #333; border-radius: 0.25rem; overflow: hidden; }
</style>
'''

# Index page template
index_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Elephant Detector</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0/dist/css/bootstrap.min.css" rel="stylesheet">
  ''' + shared_css + '''
</head>
<body>
  <div class="container py-5">
    <h1 class="mb-4">Elephant Detector</h1>
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <div class="mb-3">
        <input type="file" name="file" class="form-control bg-dark text-light" required>
      </div>
      <button type="submit" class="btn btn-primary">Detect</button>
      <a href="{{ url_for('history') }}" class="btn btn-link ms-3">History</a>
    </form>
    <div id="progress-container">
      <progress id="uploadProgress" max="100" value="0"></progress>
      <span id="progressText">0%</span>
    </div>
  </div>

  <script>
    const form = document.getElementById('upload-form');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const fd = new FormData(form);
      const res = await fetch('/', { method: 'POST', body: fd });
      if (!res.ok) { alert('Upload failed'); return; }
      const { report_id } = await res.json();
      document.getElementById('progress-container').style.display = 'block';
      const interval = setInterval(async () => {
        const pr = await fetch(`/progress/${report_id}`);
        const { progress } = await pr.json();
        document.getElementById('uploadProgress').value = progress;
        document.getElementById('progressText').innerText = progress + '%';
        if (progress >= 100) {
          clearInterval(interval);
          window.location = `/report/${report_id}`;
        }
      }, 500);
    });
  </script>
</body>
</html>
'''

# History page template
history_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>History</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0/dist/css/bootstrap.min.css" rel="stylesheet">
  ''' + shared_css + '''
</head>
<body>
  <div class="container py-5">
    <h1 class="mb-4">Detection History</h1>
    <a href="{{ url_for('index') }}" class="btn btn-link">Home</a>
    <a href="{{ url_for('export_csv') }}" class="btn btn-secondary float-end">Export CSV</a>
    <table class="table table-striped mt-3">
      <thead>
        <tr><th>ID</th><th>Time</th><th>Type</th><th>View</th></tr>
      </thead>
      <tbody>
        {% for r in reports %}
        <tr>
          <td>{{ r.id }}</td>
          <td>{{ r.timestamp }}</td>
          <td>{{ r.media_type }}</td>
          <td><a href="{{ url_for('detail', report_id=r.id) }}">Detail</a></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
'''

# Detail page template
detail_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Report {{ report.id }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0/dist/css/bootstrap.min.css" rel="stylesheet">
  ''' + shared_css + '''
</head>
<body>
  <div class="container py-5">
    <h1 class="mb-4">Report {{ report.id }}</h1>
    <a href="{{ url_for('history') }}" class="btn btn-link mb-3">Back</a>

    <h3>Original</h3>
    {% if report.media_type == 'image' %}
      <img src="{{ url_for('static', filename=report.original_path) }}" class="img-fluid mb-3 rounded">
    {% else %}
      <div class="video-container">
        <video id="origVid" controls muted>
          <source src="{{ url_for('static', filename=report.original_path) }}" type="video/{{ report.original_path.split('.')[-1] }}">
        </video>
      </div>
    {% endif %}

    <h3>Processed</h3>
    {% if report.media_type == 'image' %}
      <img src="{{ url_for('static', filename=report.processed_path) }}" class="img-fluid mb-3 rounded">
    {% else %}
      <div class="video-container">
        <video id="procVid" controls muted>
          <source src="{{ url_for('static', filename=report.processed_path) }}" type="video/webm">
        </video>
      </div>
    {% endif %}

    <div class="d-flex justify-content-between mb-4">
      <button id="playBoth" class="btn btn-primary">Play Both</button>
      <a href="{{ url_for('export_details', report_id=report.id) }}" class="btn btn-secondary">Download Details</a>
    </div>

    <h3>Details</h3>
    <p><strong>Processing Time:</strong> {{ details.processing_time | round(2) }} s</p>
    <p><strong>Average Confidence:</strong>
       {% if details.average_confidence is not none %}
         {{ details.average_confidence | round(3) }}
       {% else %}
         N/A
       {% endif %}
    </p>
    <p><strong>Max Confidence:</strong>
       {% if details.max_confidence is not none %}
         {{ details.max_confidence | round(3) }}
       {% else %}
         N/A
       {% endif %}
    </p>

    <h3>Detection Points</h3>
    <table class="table table-striped">
      <thead>
        <tr><th>ID</th><th>X1</th><th>Y1</th><th>X2</th><th>Y2</th><th>Confidence</th></tr>
      </thead>
      <tbody>
        {% for det in details.detections %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ det.x1 }}</td>
          <td>{{ det.y1 }}</td>
          <td>{{ det.x2 }}</td>
          <td>{{ det.y2 }}</td>
          <td>{{ det.confidence | round(3) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <script>
    window.addEventListener('load', () => {
      const orig = document.getElementById('origVid');
      const proc = document.getElementById('procVid');
      if (orig && proc) { orig.play(); proc.play(); }
      document.getElementById('playBoth').onclick = () => {
        if (orig) orig.play();
        if (proc) proc.play();
      };
    });
  </script>
</body>
</html>
'''

# Helper: detect on image
def detect_image(input_path, output_path):
    results = model(input_path)[0]
    img = cv2.imread(input_path)
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        dets.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': conf
        })
        cv2.rectangle(img, (x1, y1), (x2, y2), (34, 87, 255), 2)
        cv2.putText(
            img, f"{conf:.2f}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (34, 87, 255), 2, cv2.LINE_AA
        )
    cv2.imwrite(output_path, img)
    return dets

# Background processing
def process_file(report_id, fname, ext):
    start_time = time.time()
    in_abs = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    all_dets = []

    # IMAGE
    if ext in IMAGE_EXTS:
        PROGRESS[report_id] = 20
        processed_fname = f"processed_{fname}"
        out_abs = os.path.join(app.config['PROCESSED_FOLDER'], processed_fname)
        dets = detect_image(in_abs, out_abs)
        all_dets = dets

    # VIDEO
    else:
        PROGRESS[report_id] = 10
        base = os.path.splitext(fname)[0]
        cap = cv2.VideoCapture(in_abs)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Transcode for browser
        viewable_fname = f"viewable_{base}.webm"
        vw = cv2.VideoWriter(
            os.path.join(app.config['UPLOAD_FOLDER'], viewable_fname),
            fourcc, fps, (w, h)
        )
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            vw.write(frame)
            PROGRESS[report_id] = 10 + int(i / total * 20)
        cap.release()
        vw.release()

        # Detection
        cap = cv2.VideoCapture(in_abs)
        out_fname = f"processed_{base}.webm"
        vw2 = cv2.VideoWriter(
            os.path.join(app.config['PROCESSED_FOLDER'], out_fname),
            fourcc, fps, (w, h)
        )
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame)[0]
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                all_dets.append({
                    'frame': i, 'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2, 'confidence': conf
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 87, 255), 2)
                cv2.putText(
                    frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (34, 87, 255), 2, cv2.LINE_AA
                )
            vw2.write(frame)
            PROGRESS[report_id] = 30 + int(i / total * 70)
        cap.release()
        vw2.release()

    # Compute metrics
    processing_time = time.time() - start_time
    confs = [d['confidence'] for d in all_dets]
    # average for video always, for image only if >2 detections
    if confs and (ext not in IMAGE_EXTS or len(confs) > 2):
        avg_conf = sum(confs) / len(confs)
    else:
        avg_conf = None
    max_conf = max(confs) if confs else None

    details = {
        'processing_time': processing_time,
        'average_confidence': avg_conf,
        'max_confidence': max_conf,
        'detections': all_dets
    }

    # Save to DB
    with app.app_context():
        r = Report.query.get(report_id)
        if ext in IMAGE_EXTS:
            r.processed_path = f"{PROCESSED_SUBFOLDER}/{processed_fname}"
        else:
            r.original_path = f"{UPLOAD_SUBFOLDER}/{viewable_fname}"
            r.processed_path = f"{PROCESSED_SUBFOLDER}/{out_fname}"
        r.details = json.dumps(details)
        db.session.commit()

    PROGRESS[report_id] = 100

# Routes

@app.route('/', methods=['GET'])
def index():
    return render_template_string(index_template)

@app.route('/', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file'}), 400

    fname = secure_filename(file.filename)
    ext = fname.rsplit('.', 1)[1].lower()
    abs_in = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(abs_in)

    media_type = 'image' if ext in IMAGE_EXTS else 'video'
    r = Report(
        media_type=media_type,
        original_path=f"{UPLOAD_SUBFOLDER}/{fname}"
    )
    db.session.add(r)
    db.session.commit()

    PROGRESS[r.id] = 0
    threading.Thread(
        target=process_file,
        args=(r.id, fname, ext),
        daemon=True
    ).start()

    return jsonify({'report_id': r.id})

@app.route('/progress/<int:report_id>')
def progress(report_id):
    return jsonify({'progress': PROGRESS.get(report_id, 0)})

@app.route('/history')
def history():
    reports = Report.query.order_by(Report.timestamp.desc()).all()
    return render_template_string(history_template, reports=reports)

@app.route('/report/<int:report_id>')
def detail(report_id):
    r = Report.query.get_or_404(report_id)
    details = json.loads(r.details)
    return render_template_string(detail_template,
                                  report=r,
                                  details=details)

@app.route('/export')
def export_csv():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'id', 'timestamp', 'media_type',
        'original_path', 'processed_path', 'details'
    ])
    for rep in Report.query.all():
        writer.writerow([
            rep.id,
            rep.timestamp,
            rep.media_type,
            rep.original_path,
            rep.processed_path,
            rep.details
        ])
    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=reports.csv'}
    )

@app.route('/export_details/<int:report_id>')
def export_details(report_id):
    r = Report.query.get_or_404(report_id)
    details = json.loads(r.details)
    payload = json.dumps(details, indent=2)
    return Response(
        payload,
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment;filename=details_{report_id}.json'}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
