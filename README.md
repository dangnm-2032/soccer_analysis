# soccer_analysis

## Prequiste
- Python 3.9

## Install
```bash
git clone --recurse-submodules https://github.com/dangnm-2032/soccer_analysis.git

cd soccer_analysis

Download https://drive.google.com/file/d/1bSUNpvMfJkvCFOu-TK-o7iGY1p-9BxmO/view?usp=sharing and place to components/centroids_reid/models

Download https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe and place to components/ViTPose/checkpoints

Download https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing and place to components/models

Download https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing and place to components/models

Get yolo-based detector model and place to models/

python3 -m .env

source .env/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python app.py
```

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