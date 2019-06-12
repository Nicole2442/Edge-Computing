#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
created by Nicole Liao
'''

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

model_xml = "op_32.xml"
model_bin = "op_32.bin"
image_test = "your_image.jpeg"

def main():
    # device initialization
    plugin = IEPlugin(device="GPU") # device="MYRIAD"

    # load neural networks with IR file
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)

    # Read and pre-process input images
    image = cv2.imread(image_test)
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    # start inference
    log.info("Starting inference")
    t0 = time()
    res = exec_net.infer(inputs={input_blob: image})
    infer_time = (time() - t0) * 1000
    print("running time: %f ms", infer_time)

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("results: ")

    # clean up
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    main()
