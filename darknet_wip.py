from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import Dict, List, Tuple


def parse_cfg(cfg_file: str) -> List[Dict]:
    """ Read the config file and append every block as a key-value pair to a list

    Example config block:
    [shortcut]
    from=-3
    activation=linear
    """
    file = open(cfg_file, "r")
    lines = file.read().split("\n")  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # remove empty lines
    lines = [x for x in lines if x[0] != "#"]  # remove commented lines
    lines = [x.rstrip().lstrip() for x in lines]  # trim whitespace

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # start of a new block
            if len(block) != 0:  # not empty line
                blocks.append(block)  # append an empty dict
                block = {}
            block["type"] = line[1:-1].rstrip()  # name the block type
        else:
            key, val = line.split("=")  # append the block w/ a parameter
            block[key.rstrip()] = val.lstrip()

    blocks.append(block)

    return blocks


class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(torch.nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(cfg_blocks: List[Dict]) -> Tuple[Dict, torch.nn.ModuleList]:
    net_info = cfg_blocks[0]
    module_list = torch.nn.ModuleList()
    prev_filters = 3  # we need to keep track of number of filters over time
    output_filters = []

    # for each block in the list, create a PyTorch module
    for index, block in enumerate(cfg_blocks[1:]):
        seq_module = torch.nn.Sequential()
        block_type = block["type"]

        if block_type == "convolutional":
            activation_layer = block["activation"]
            batch_norm = int(block.get("batch_normalize", 0))
            bias = batch_norm == 0

            filters = int(block["filters"])
            padding_flag = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            pad = (kernel_size - 1) // 2 if padding_flag else 0

            conv = torch.nn.Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias
            )
            seq_module.add_module(f"conv_{index}", conv)

            if batch_norm:
                batch_norm_layer = torch.nn.BatchNorm2d(filters)
                seq_module.add_module(f"batch_norm_{index}", batch_norm_layer)

            if activation_layer == "leaky":
                leaky_relu_module = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
                seq_module.add_module(f"leaky_relu_{index}", leaky_relu_module)

        elif block_type == "upsample":
            upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
            seq_module.add_module(f"upsample_{index}", upsample)

        elif block_type == "route":
            block["layers"] = block["layers"].split(",")
            start = int(block["layers"][0])
            end = int(block["layers"][1]) if len(block["layers"]) > 1 else 0

            if start > 0:
                start -= index
            if end > 0:
                end -= index

            route = EmptyLayer()  # dummy layer + perform the concatentation in the forward module
            seq_module.add_module(f"route_{index}", route)

            # update the filters to hold the number of filters output by a route layer
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif block_type == "shortcut":
            seq_module.add_module(f"shortcut_{index}", EmptyLayer())

        elif block_type == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = [int(a) for a in block["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            seq_module.add_module(f"Detection_{index}", DetectionLayer(anchors))

        module_list.append(seq_module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class Darknet(torch.nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, block, CUDA):
        modules = self.blocks[1:]  # first element of blocks is a "net" block
        outputs = {}


blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
