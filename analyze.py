from components.utils import read_video, save_video
from components.trackers import Tracker
from components.team_assigner import TeamAssigner
from components.player_ball_assigner import PlayerBallAssigner
from components.camera_movement_estimator import CameraMovementEstimator
from components.view_transformer import ViewTransformer
from components.speed_and_distance_estimator import SpeedAndDistance_Estimator

import components.configuration as config
import components.helpers as helpers
import components.legibility_classifier as lc

import numpy as np
import argparse
import os
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import pickle
import shutil

def get_soccer_net_legibility_results(args, use_filtered = False, filter = 'sim', exclude_balls=True):
    root_dir = config.dataset['root_dir']
    image_dir = config.dataset['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = os.listdir(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['working_dir'],
                                                  config.dataset['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['working_dir'],
                                                  config.dataset['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['working_dir'],
                                        config.dataset['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets


    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['legibility_model'], arch=config.dataset['legibility_model_arch'], threshold=0.5)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['working_dir'], config.dataset['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['working_dir'], config.dataset['illegible_result'])
    json_object = json.dumps({'illegible': illegible_tracklets}, indent=4)
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json_object)

    return legible_tracklets, illegible_tracklets

def generate_json_for_pose_estimator(args, legible = None):
    all_files = []
    if not legible is None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['root_dir'])
        image_dir = config.dataset['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = os.listdir(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = os.listdir(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['working_dir'], config.dataset['pose_input_json'])
    helpers.generate_json(all_files, output_json)

def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if not str(entry) in dict.keys():
            dict[str(entry)] = -1

    all_tracks = os.listdir(image_dir)
    for t in all_tracks:
        if not t in dict.keys():
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict        

def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    Path(config.dataset['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True
    current_dir = os.getcwd()
    image_dir = os.path.join(current_dir, config.dataset['root_dir'], config.dataset['images'])
    soccer_ball_list = os.path.join(config.dataset['working_dir'],
                                      config.dataset['soccer_ball_list'])
    features_dir = os.path.join(current_dir, config.dataset['feature_output_folder'])
    full_legibile_path = os.path.join(config.dataset['working_dir'],
                                      config.dataset['legible_result'])
    illegible_path = os.path.join(config.dataset['working_dir'],
                                  config.dataset['illegible_result'])
    input_json = os.path.join(config.dataset['working_dir'],
                              config.dataset['pose_input_json'])
    output_json = os.path.join(config.dataset['working_dir'],
                               config.dataset['pose_output_json'])

    # 1. Filter out soccer ball based on images size
    if args.pipeline['soccer_ball_filter']:
        print("Determine soccer ball")
        success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done determine soccer ball")

    # 1. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        print("Generate features")
        _env_path = os.path.join(current_dir, config.env)
        command = f"cd components && {_env_path}/bin/python {config.reid_script} --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done generating features")

    #2. identify and remove outliers based on features
    if args.pipeline['filter'] and success:
        print("Identify and remove outliers")
        command = f".env/bin/python3 components/gaussian_outliers.py --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done removing outliers")

    #3. pass all images through legibililty classifier and record results
    if args.pipeline['legible'] and success:
        print("Classifying Legibility:")
        try:
            legible_dict, illegible_tracklets = get_soccer_net_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=True)
        except Exception as error:
            print(f'Failed to run legibility classifier:{error}')
            success = False
        print("Done classifying legibility")

    #4. generate json for pose-estimation
    if args.pipeline['pose'] and success:
        print("Generating json for pose")
        try:
            if legible_dict is None:
                with open(full_legibile_path, 'r') as openfile:
                    # Reading from json file
                    legible_dict = json.load(openfile)
            generate_json_for_pose_estimator(args, legible = legible_dict)
        except Exception as e:
            print(e)
            success = False
        print("Done generating json for pose")

        #5. run pose estimation and store results
        if success:
            print("Detecting pose")
            command = f".env/bin/python3 components/pose.py {config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
                {config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} \
                --out-json {output_json}"
            success = os.system(command) == 0
            print("Done detecting pose")


    #6. generate cropped images
    if args.pipeline['crops'] and success:
        print("Generate crops")
        try:
            crops_destination_dir = os.path.join(config.dataset['working_dir'], config.dataset['crops_folder'], 'imgs')
            Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
            if legible_results is None:
                with open(full_legibile_path, "r") as outfile:
                    legible_results = json.load(outfile)
            helpers.generate_crops(output_json, crops_destination_dir, legible_results)
        except Exception as e:
            print(e)
            success = False
        print("Done generating crops")

    str_result_file = os.path.join(config.dataset['working_dir'],
                                   config.dataset['jersey_id_result'])
    #7. run STR system on all crops
    if args.pipeline['str'] and success:
        print("Predict numbers")
        image_dir = os.path.join(config.dataset['working_dir'], config.dataset['crops_folder'])

        command = f".env/bin/python3 components/str.py  {config.dataset['str_model']}\
            --data_root={image_dir} --batch_size=1 --inference --result_file {str_result_file}"
        success = os.system(command) == 0
        print("Done predict numbers")

    #str_result_file = os.path.join(config.dataset['working_dir'], "val_jersey_id_predictions.json")
    if args.pipeline['combine'] and success:
        #8. combine tracklet results
        results_dict, analysis_results = helpers.process_jersey_id_predictions(str_result_file, useBias=True)
        # add illegible tracklet predictions
        consolidated_dict = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=soccer_ball_list)

        #save results as json
        final_results_path = os.path.join(config.dataset['working_dir'], config.dataset['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)


def main(args, video_path):
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker('models/detector.pt')
    tracks = tracker.get_object_tracks(video_frames)

    # Get player number

    actions = {
        "soccer_ball_filter": True,
        "feat": True,
        "filter": True,
        "legible": True,
        "pose": True,
        "crops": True,
        "str": True,
        "combine": True,
    }
    args.pipeline = actions
    try:
        soccer_net_pipeline(args)
        result_path = 'temp/final_results.json'
        with open(result_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error in soccer_net_pipeline: {e}")
        data = {}
        raise e

    for frame_num, frame_data in enumerate(tracks['players']):
        for track_id in frame_data:
            if str(track_id) in data:
                tracks['players'][frame_num][track_id]['number'] = int(data[str(track_id)])
            else:
                tracks['players'][frame_num][track_id]['number'] = -1
        
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    try:
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    except:
        None
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    with open(f"results/{args.job_id}.pkl", "wb") as f:
        pickle.dump((tracks, team_ball_control, camera_movement_per_frame), f)
    
    # Delete temp files
    shutil.rmtree('temp')
    if os.path.exists(video_path):
        os.remove(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path to the video file")
    parser.add_argument('job_id', help="Job ID")
    args = parser.parse_args()
    
    video_path = args.video_path
    
    main(args,video_path)