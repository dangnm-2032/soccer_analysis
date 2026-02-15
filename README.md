# soccer_analysis

## Prequiste
- Python 3.9

## Install
- python3 -m .env
- source .env/bin/activate
- pip install -r requirements.txt

## Run
- python app.py

## Usage
### Upload video to analyze
```bash
curl -X POST http://localhost:5000/analyze -F "file=@path/to/your/video.mp4
```
### Check status
```bash
curl http://localhost:5000/status/{job_id}
```
### Get result
```bash
curl -O http://localhost:5000/result/{job_id}
```
It will return a pickle file, to load it, use 

```python
import pickle

with open(f"{job_id}.pkl", "rb") as f:
    tracks, team_ball_control, camera_movement_per_frame = pickle.load(f)
```



## Thanks to
- https://github.com/abdullahtarek/football_analysis
- https://github.com/mkoshkina/jersey-number-pipeline