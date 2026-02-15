from flask import Flask, request, jsonify, send_file
import uuid
import threading
import queue
import time
import os

app = Flask(__name__)

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
        job_id, video_path = job_queue.get()
        try:
            jobs[job_id]["status"] = "PROCESSING"

            # ---- Simulate heavy video analysis ----
            command = f"python analyze.py {video_path} {job_id}"
            success = os.system(command) == 0
            if not success:
                raise Exception("Analysis failed")

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
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{video.filename}")
    video.save(video_path)

    jobs[job_id] = {
        "status": "PENDING",
        "result": None,
        "error": None,
    }

    job_queue.put((job_id, video_path))

    return jsonify({
        "queue_id": job_id,
        "status": "PENDING"
    }), 202


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

    file_path = os.path.join("results", f"{job_id}.pkl")
    return send_file(
        file_path,
        as_attachment=True,     # forces download
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
