"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import pickle
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork
from animate import get_animation_region_params
# import matplotlib
#
# matplotlib.use('Agg')
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          **config['model_params']['generator_params'])
    if not cpu:
        generator.cuda()

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if not cpu:
        avd_network.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()

    return generator, region_predictor, avd_network


def make_animation(source_image, driving_video, generator, region_predictor, avd_network,
                   animation_mode='standard', cpu=False):

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        source_region_params = region_predictor(source)
        driving_region_params_initial = region_predictor(driving[:, :, 0])

        # source_heatmap = source_region_params['heatmap']
        # source_heatmap = F.interpolate(source_heatmap, size=(256,256))
        # source_heatmap = np.transpose(source_heatmap.data.numpy(), [0, 2, 3, 1])
        # heatmap_color = draw_colored_heatmap(source_heatmap, plt.get_cmap('gist_rainbow'), (0,0,0) ).squeeze()
        #
        # heatmap_im = Image.fromarray((heatmap_color* 255).astype(np.uint8))
        # heatmap_im.save("source_heatmap.jpeg")
        # exit()

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_region_params = region_predictor(driving_frame)
            new_region_params = get_animation_region_params(source_region_params, driving_region_params,
                                                            driving_region_params_initial, avd_network=avd_network,
                                                            mode=animation_mode)

            out = generator(source, source_region_params=source_region_params, driving_region_params=new_region_params)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

            # heatmap = driving_region_params['heatmap']
            # heatmap = F.interpolate(heatmap, size=(256,256))
            # heatmap = np.transpose(heatmap.data.numpy(), [0, 2, 3, 1])
            # heatmap_color = draw_colored_heatmap(heatmap, plt.get_cmap('gist_rainbow'), (0,0,0) ).squeeze()
            #
            # heatmap_im = Image.fromarray((heatmap_color* 255).astype(np.uint8))
            # heatmap_im.save("driving_heatmaps/driving_{:03}.jpeg".format(frame_idx))


    return predictions


def main(opt):
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    reader.close()
    driving_video = imageio.mimread(opt.driving_video, memtest=False)

    source_image = resize(source_image, opt.img_shape)[..., :3]
    # print (source_image)
    # s_im = Image.fromarray((source_image* 255).astype(np.uint8))
    # s_im.save("OOD_Sample.png")
    # exit()
    # driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    # with open('driving_video.pickle', 'wb') as f:
    #     pickle.dump(driving_video, f)

    with open('driving_video.pickle', 'rb') as f:
        driving_video = pickle.load(f)

    generator, region_predictor, avd_network = load_checkpoints(config_path=opt.config,
                                                                checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(source_image, driving_video, generator, region_predictor, avd_network,
                                 animation_mode='avd', cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='ted384.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--mode", default='avd', choices=['standard', 'relative', 'avd'],
                        help="Animation mode")
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
