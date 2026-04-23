import os
import shutil 
import argparse
import yaml
import subprocess
import requests
import zipfile
import pickle
import pandas as pd

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

def main(job_id, video_path, callback_url, video_id, csv_path):
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
    
    # Get team ball possession
    file_path = os.path.join(
        "components","sn-gamestate","outputs",
        job_id,"states","sn-gamestate.pklz"
    )

    with zipfile.ZipFile(file_path, "r") as z:
        for name in z.namelist():
            if '.pkl' in name and "image" not in name:
                with z.open(name) as f:
                    data = pickle.load(f)
                break

    team_ball_possesion = data.iloc[-1][['frame_count', 'possession_left', 'possession_right']].to_dict()

    df = pd.read_csv(csv_path, skiprows=2)
    meta = pd.read_csv(csv_path, nrows=1).columns.tolist()
    meta_vals = pd.read_csv(csv_path, skiprows=0, nrows=1).iloc[0].to_dict()
    speed_map = data.groupby("jersey_number")["speed"].mean()
    df["avg_speed"] = df["shirt_number"].astype(str).map(speed_map)
    possession_percent = (
        data.assign(
            jersey_filled=lambda df: df["jersey_number"].fillna(
                df["track_id"].map(
                    df.dropna(subset=["jersey_number"])
                    .groupby("track_id")["jersey_number"]
                    .agg(lambda x: x.value_counts().idxmax())
                )
            )
        )
        .dropna(subset=["jersey_filled"])
        .loc[lambda df: df["ball_control"]]
        .groupby("jersey_filled")
        .size()
        .pipe(lambda s: s / s.sum() * 100)
        .reset_index(name="possession_percent")
    )
    df["shirt_number"] = pd.to_numeric(df["shirt_number"], errors="coerce")
    possession_percent["jersey_filled"] = pd.to_numeric(possession_percent["jersey_filled"], errors="coerce")
    result = df.merge(
        possession_percent,
        left_on="shirt_number",
        right_on="jersey_filled",
        how="left"
    ).drop(columns=["jersey_filled"]).fillna(0)
    meta_df = pd.DataFrame([meta_vals])[meta]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        meta_df.to_csv(f, index=False)
        result.to_csv(f, index=False)

    headers = {
        'Content-Type': 'application/json',
    }

    json_data = {
        'job_id': job_id,
        'video_path': f"http://127.0.0.1:5000/videos/{job_id}.mp4",
        'video_id': video_id,
        'possession_left': int(team_ball_possesion['possession_left'] / team_ball_possesion['frame_count'] * 100),
        'possession_right': int(team_ball_possesion['possession_right'] / team_ball_possesion['frame_count'] * 100)
        
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
    parser.add_argument('csv_path', help="CSV path")
    args = parser.parse_args()
    
    video_path = args.video_path
    job_id = args.job_id
    callback_url = args.callback_url
    video_id = args.video_id
    csv_path = args.csv_path
    
    main(job_id, video_path, callback_url, video_id, csv_path)