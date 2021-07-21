"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, pixelwise_flow_predictor_params=None, skips=False, revert_axis_swap=True):
        super(Generator, self).__init__()

        if pixelwise_flow_predictor_params is not None:
            self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_channels=num_channels,
                                                                   revert_axis_swap=revert_axis_swap,
                                                                   **pixelwise_flow_predictor_params)
        else:
            self.pixelwise_flow_predictor = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.skips = skips

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out

    def forward(self, source_image, driving_region_params, source_region_params, bg_params=None):
        out = self.first(source_image)
        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        output_dict = {}
        if self.pixelwise_flow_predictor is not None:
            motion_params = self.pixelwise_flow_predictor(source_image=source_image,
                                                          driving_region_params=driving_region_params,
                                                          source_region_params=source_region_params,
                                                          bg_params=bg_params)
            output_dict["deformed"] = self.deform_input(source_image, motion_params['optical_flow'])
            if 'occlusion_map' in motion_params:
                output_dict['occlusion_map'] = motion_params['occlusion_map']
        else:
            motion_params = None

        # flow = motion_params['optical_flow'][0].squeeze()
        # print (flow.shape)
        # print (flow)
        # flow_np = flow.detach().cpu().numpy()
        # flow_im = flow_vis.flow_to_color(flow_np, convert_to_bgr=False)
        # print (flow_im)
        # flow_im = Image.fromarray(flow_im)
        # flow_im.save('flow.png')
        # exit()

        # driving_heatmap = F.interpolate(driving_region_params['heatmap'] , size=(256,256) )
        # driving_heatmap = np.transpose(driving_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
        # # driving_heatmap = driving_heatmap[0].squeeze()
        # driving_heatmap_out = draw_colored_heatmap(driving_heatmap, plt.get_cmap('gist_rainbow'), np.array((0,0,0)))
        #
        # source_heatmap = F.interpolate(source_region_params['heatmap'], size=(256,256) )
        # source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
        # # source_heatmap = source_heatmap[0].squeeze()
        # source_heatmap_out = draw_colored_heatmap(source_heatmap, plt.get_cmap('gist_rainbow'), np.array((0,0,0)))
        #
        # src_hm = (255 * source_heatmap_out[0] ).astype(np.uint8)
        # drv_hm = (255 * driving_heatmap_out[0] ).astype(np.uint8)
        # print ( src_hm.shape, drv_hm.shape)
        # src_hm_im = Image.fromarray(src_hm)
        # drv_hm_im = Image.fromarray(drv_hm)
        # src_hm_im.save('src_hm.png')
        # drv_hm_im.save('drv_hm.png')
        #
        # exit()


        out = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            if self.skips:
                out = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = self.up_blocks[i](out)
        if self.skips:
            out = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out = self.final(out)
        out = F.sigmoid(out)


        if self.skips:
            out = self.apply_optical(input_skip=source_image, input_previous=out, motion_params=motion_params)

        output_dict["prediction"] = out

        return output_dict
