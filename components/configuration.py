env = '.env/'

pose_home = 'components/ViTPose'

str_home = 'str/parseq/'

# centroids
reid_script = 'centroid_reid.py'

dataset = {
    'root_dir': './temp/cropped_objects/',
    'working_dir': './temp',

    'images': 'players',
    'feature_output_folder': './temp/features',
    'illegible_result': 'illegible.json',
    'soccer_ball_list': 'soccer_ball.json',
    'sim_filtered': 'features/main_subject_0.4.json',
    'gauss_filtered': 'features/main_subject_gauss_th=3.5_r=3.json',
    'legible_result': 'legible.json',
    'raw_legible_result': 'raw_legible_resnet34.json',
    'pose_input_json': 'pose_input.json',
    'pose_output_json': 'pose_results.json',
    'crops_folder': 'crops',
    'jersey_id_result': 'jersey_id_results.json',
    'final_result': 'final_results.json',

    'numbers_data': 'lmdb',

    'legibility_model': "components/models/legibility_resnet34_soccer_20240215.pth",
    'legibility_model_arch': "resnet34",

    'legibility_model_url':  "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
    'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
    'str_model': 'components/models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',

    #'str_model': 'pretrained=parseq',
    'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
}