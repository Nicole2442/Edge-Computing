#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
created by Nicole Liao @2019/5/12
'''

import tensorflow as tf

model_file = "op.pb"
# if NCS: ['input_1','input_2','input_3']
# if TPU Edge: ['input_1']
input_arrays = ['input_1','input_2','input_3'] 
output_arrays = ['Mconv7_stage2_L1/BiasAdd','Mconv8_stage2_L1/BiasAdd']
# if NCS: {'input_1':(1,368,368,3),'input_2':(1,46,46,38),'input_3':(1,46,46,19)}
# if TPU Edge: {'input_1':(1,368,368,3)}
input_shapes = {'input_1':(1,368,368,3)} 

converter = tf.lite.TFLiteConverter.from_frozen_graph(model_file, input_arrays, output_arrays, input_shapes)
tflite_model = converter.convert()
open("op.tflite", "wb").write(tflite_model)
