from flask import Flask, request, jsonify, send_file, send_from_directory, Response
import uuid
import threading
import queue
import time
import subprocess
import os
import re
from flask_cors import CORS

from analyze import main

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# In-memory stores
job_queue = queue.Queue()
jobs = {}  
# jobs[job_id] = {
#   "status": "PENDING" | "PROCESSING" | "DONE" | "FAILED",
#   "result": None,
#   "error": None
# }

# ---------------------------
# Background worker
# ---------------------------
def worker():
    while True:
        job_id, video_path, callback_url, video_id, csv_path = job_queue.get()
        try:
            jobs[job_id]["status"] = "PROCESSING"

            # ---- Simulate heavy video analysis ----
            main(job_id, video_path, callback_url, video_id, csv_path)

            jobs[job_id]["status"] = "DONE"

        except Exception as e:
            jobs[job_id]["status"] = "FAILED"
            jobs[job_id]["error"] = str(e)

        finally:
            job_queue.task_done()


threading.Thread(target=worker, daemon=True).start()

# ---------------------------
# Routes
# ---------------------------
@app.route("/analyze", methods=["POST"])
def analyze_video():
    data = request.get_json()

    # ❌ Không còn request.files nữa
    if not data or "video_path" not in data:
        return jsonify({"error": "No video_path provided"}), 400

    video_path = data["video_path"]
    csv_path = data['csv_path']

    # Validate path
    if not isinstance(video_path, str) or video_path.strip() == "":
        return jsonify({"error": "Invalid video_path"}), 400

    # ❗ Check file tồn tại
    if not os.path.exists(video_path):
        return jsonify({"error": "File does not exist"}), 400

    if not data or "callback_url" not in data:
        return jsonify({"error": "No callback_url provided"}), 400

    callback_url = data["callback_url"]
    video_id = data["video_id"]
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "PENDING",
        "result": None,
        "error": None,
    }

    # 👇 push vào queue như cũ
    job_queue.put((job_id, video_path, callback_url, video_id, csv_path))

    return jsonify({
        "queue_id": job_id,
        "status": "PENDING"
    }), 202

@app.route("/videos/<filename>")
def stream_video(filename):
    video_path = os.path.join("results", filename)

    range_header = request.headers.get("Range", None)
    if not range_header:
        return send_file(video_path, mimetype="video/mp4")

    size = os.path.getsize(video_path)
    byte1, byte2 = 0, None

    match = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if match:
        byte1 = int(match.group(1))
        if match.group(2):
            byte2 = int(match.group(2))

    length = size - byte1 if byte2 is None else byte2 - byte1 + 1

    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4')
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))

    return rv
@app.route("/status/<job_id>", methods=["GET"])
def check_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid queue id"}), 404

    return jsonify({
        "queue_id": job_id,
        "status": job["status"]
    })


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid queue id"}), 404

    if job["status"] != "DONE":
        return jsonify({
            "queue_id": job_id,
            "status": job["status"],
            "message": "Result not ready"
        }), 202

    file_path = os.path.join("components/sn-gamestate/outputs", job_id, "states", "sn-gamestate.pklz")
    return send_file(
        file_path,
        as_attachment=True,     # forces download
        download_name=job_id + ".pklz"
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
