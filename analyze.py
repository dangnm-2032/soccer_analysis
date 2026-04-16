import os
import shutil 
import argparse
import yaml
import subprocess
import requests

CONFIG_PATH = 'components/sn-gamestate/sn_gamestate/configs/soccernet.yaml'
EXTRACTED_FRAME_PATH = 'components/sn-gamestate/data/Analyze/valid'

def update_config(job_id):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['job_jd'] = job_id
    config['dataset']['vids_dict']['valid'] = [job_id]
    config['hydra']['run']['dir'] = "outputs/${job_jd}"
    config['dataset']['dataset_path'] = r"${data_dir}/Analyze"
    config['eval_tracking'] = False
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

def extract_frames(video_path, job_id):
    output_dir = os.path.join(EXTRACTED_FRAME_PATH, job_id, 'img1')

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-qscale:v", "1",
        f"{output_dir}/%06d.jpg"
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

def convert_h264(input_path, output_path):
    command = [
        "ffmpeg", "-i", input_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-movflags", "+faststart", "-c:a", "aac", output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

def run():
    command = "cd components/sn-gamestate && uv run tracklab -cn soccernet"
    subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL)

def main(job_id, video_path, callback_url, video_id):
    update_config(job_id)
    extract_frames(video_path, job_id)
    
    run()

    # Delete temp files
    shutil.rmtree(os.path.join(EXTRACTED_FRAME_PATH, job_id))
    if os.path.exists(video_path):
        os.remove(video_path)

    file_path = os.path.join("components/sn-gamestate/outputs", job_id, "visualization", "videos", f"{job_id}.mp4")
    
    # move file to results folder
    os.makedirs("results", exist_ok=True)
    
    # convert to h264
    convert_h264(file_path, os.path.join("results", f"{job_id}.mp4"))
    
    headers = {
        'Content-Type': 'application/json',
    }

    json_data = {
        'job_id': job_id,
        'video_path': f"http://0.0.0.0:5000/videos/{job_id}.mp4",
        'video_id': video_id
    }

    response = requests.post(
        callback_url, 
        headers=headers, 
        json=json_data
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path to the video file")
    parser.add_argument('job_id', help="Job ID")
    parser.add_argument('callback_url', help="Callback URL")
    parser.add_argument('video_id', help="Video ID")
    args = parser.parse_args()
    
    video_path = args.video_path
    job_id = args.job_id
    callback_url = args.callback_url
    video_id = args.video_id
    
    main(job_id, video_path, callback_url)