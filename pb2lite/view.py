#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
created by Nicole Liao @2019/5/12
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
model = "op.pb"

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWrite = tf.summary.FileWriter('log/', graph)
    
