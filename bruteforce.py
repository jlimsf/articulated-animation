import os
import subprocess
import imageio
from skimage.transform import resize
import pickle


src_img_path = '../clean_rebecca_taylor_frontals'
driving_train_vids = 'data/fashion_vids/train'
driving_test_vids = 'data/fashion_vids/test'
bruteforce_result_path = 'bruteforce_results/'
pickle_vids = 'data/fashion_vids_pickle'

#create pickle versions for each video so we don't have to keep reading them
ct = 0
for mode in [driving_train_vids, driving_test_vids]:

    mode_pickle_dir = os.path.join(pickle_vids, mode.split('/')[-1])
    if not os.path.exists(mode_pickle_dir):
        os.makedirs(mode_pickle_dir)

    for driving_video in os.listdir(mode):
        print ("Working on {}".format(driving_video))
        driving_vid_fp = os.path.join(mode, driving_video)

        vid_pickle_dir = os.path.join(mode_pickle_dir, driving_video.split('.')[0])
        if not os.path.exists(vid_pickle_dir):
            os.makedirs(vid_pickle_dir)

        driving_vid_pkl_file = os.path.join(vid_pickle_dir, 'vid.pickle')

        reader = imageio.get_reader(driving_vid_fp)
        fps = reader.get_meta_data()['fps']
        reader.close()

        driving_video = imageio.mimread(driving_vid_fp, memtest=False)

        loaded_video = [resize(frame, (256,256))[..., :3] for frame in driving_video]

        dat = {'video': loaded_video, 'fps': fps}
        with open(driving_vid_pkl_file, 'wb') as f:
            pickle.dump(dat, f)
        ct += 1
        if ct == 20:
            break

if not os.path.exists(bruteforce_result_path):
    os.makedirs(bruteforce_result_path)

for idx, src_im in enumerate(os.listdir(src_img_path)):

    print ("Working on Image {} out of {}".format(idx, len(os.listdir(src_img_path))))

    src_im_fp = os.path.join(src_img_path, src_im)
    src_im_dir = os.path.join(bruteforce_result_path, src_im.split('.')[0])

    if not os.path.exists(src_im_dir):
        os.makedirs(src_im_dir)

    for driving_video in os.listdir(driving_train_vids):

        driving_video_fp = os.path.join(driving_train_vids, driving_video)
        driving_video_pickle_fp = os.path.join(pickle_vids, 'train',driving_video.split('.')[0], 'vid.pickle')

        result_dir = os.path.join(src_im_dir, driving_video.split('.')[0])

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_vid_fp = os.path.join(result_dir, 'result.mp4')

        command = "python demo.py --config 'config/fashion.yaml' --checkpoint 'log/fashion 22_06_21_21.13.47/00000029-cpk-avd.pth'  --source_image {}  --driving_video {} --img_shape 256,256  --mode avd --result_video {}".format(src_im_fp, driving_video_pickle_fp, result_vid_fp)

        subprocess.call(command, shell=True)



    for driving_video in os.listdir(driving_test_vids):

        driving_video_fp = os.path.join(driving_test_vids, driving_video)
        driving_video_pickle_fp = os.path.join(pickle_vids, 'test',driving_video.split('.')[0], 'vid.pickle')
        result_dir = os.path.join(src_im_dir, driving_video.split('.')[0])

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_vid_fp = os.path.join(result_dir, 'result.mp4')

        command = "python demo.py --config 'config/fashion.yaml' --checkpoint 'log/fashion 22_06_21_21.13.47/00000029-cpk-avd.pth'  --source_image {}  --driving_video {} --img_shape 256,256  --mode avd --result_video {}".format(src_im_fp, driving_video_pickle_fp, result_vid_fp)

        subprocess.call(command, shell=True)

#for every rebecctaylor image
    #for every training video
    #for every testing video
