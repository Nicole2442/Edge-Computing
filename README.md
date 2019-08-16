# How to run your own model (Keras or Tensorflow) with Intel Movidius Neural Compute Stick and Google Coral Edge TPU USB accelerator

Most deep learning achievements need computing resources and cloud computing is the way to deal with the resource issue. While in some scenario, we care more about low latency, bandwidth, security or cost. Hosting your deep learning model on the cloud may not be the better choice in such situation.

Deep learning on the edge provide solutions to the above problems. As the computing resource is the key point for edge computing, many companies provide their own platforms to deploy the deep learning models on the normal but limited hardware platform (e.g. Raspberry Pi). We will discuss two main platforms in the following content:

1. [Intel Movidius Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick) 
2. [Google Coral Edge TPU USB accelerator](https://coral.withgoogle.com)

Based on the above links, we could easily setup the environment and pass demo test. Also the two platform provides many converted models to developers. 

While if we design own our algorithm, we need convert our own model and deploy it on the hardware platform. In the following content, we will discuss how to convert and deploy our own model (Keras or Tensorflow based) on these two edge computing platforms. 

We take the OpenPose model for example. The model we convert is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) algorithm. We accelerate this algorithm from two sides: software and hardware. For the software acceleration part, we simplify the network structure and achieve it based on Keras. 

After that, we try hardware acceleration on two platforms: [Intel Movidius Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick) and [Google Coral Edge TPU USB accelerator](https://coral.withgoogle.com).

## Convert Keras model on Intel Movidius Neural Compute Stick 

There are two ways for transitioning provided by Intel: OpenVINO toolkit and NCSDK. In this blog, we use OpenVINO toolkit. If you want to know how to use NCSDK to convert model and the difference between OpenVINO toolkit and NCSDK, more details are [here](https://software.intel.com/en-us/articles/transitioning-from-intel-movidius-neural-compute-sdk-to-openvino-toolkit).

### OpenVINO installation and test

Step 1. Follow the [tutorial](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick) to install OpenVINO toolkit. 
After installation, there are basic three folders in our home dictionary:  
xxx@xxx:your_path/intel/openvino_xxxx.x.xxx  
xxx@xxx:your_path/inference_engine_samples_build  
xxx@xxx:your_path/openvino_models

Step 2. Run demo

```markdown
cd your_path/intel/openvino_xxxx.x.xxx/deployment_tools/demo/demo_squeezenet_download_convert_run.sh
```

Note: 
the "target" and "target_precision" need preset before run .sh:
1. if use CPU: tartget="CPU" and target_precision="FP32"
2. if use GPU: tartget="GPU" and target_precision="FP32"
3. if use Neural Compute Stick: target="MYRIAD" and target_precision="FP16"

### Convert Keras model to Tensorflow model (if use Tensorflow mode, skip this step)

Step 1.Train and saving the Keras model

Keras model is the file .h5. Because the nerwork structure and weights are all need for model transitioning, we save model with setting the parameter save_weights_only as false.

```markdown
checkpoint = ModelCheckPoint(weights_best_file, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='min', period=1)
```

Step 2. Convert Keras model to Tensorflow model

We use [Keras to Tensorflow tool](https://github.com/amir-abdi/keras_to_tensorflow) for converting.

Note: as we use self-defined loss function, we need modify the keras_to_tensorflow.py for the self-defined loss function:

```markdown
# add custom loss when load model
def _eucl_loss(x,y):
    batch_size = 32
    return K.sum(K.square(x-y)) / batch_size / 2

......

    model = keras.models.load_model(input_model_path, custom_objects={'_eucl_loss':_eucl_loss})
```

After that, we convert Keras model (.h5) to tensowflow model (.pb) with the following command. The tensorfow model is the frozen graph.

```markdown
cd your_path/keras_to_tensorflow
python keras_to_tensorflow.py --input_model='xxx.h5' --output_model='yyy.pb'
```

### Use TensorBoard to check the input, output and network

```markdown
cd your_path/pb2lite
python view.py
tensorboard --logdir log/
```
Then, open the web browser and input: localhost:6006. 

In the graph, you need to check the following content:
1. Check the input name, input dim, output name and output dim  
input name: input_1 with shape [1,368,368,3]  
 ![image](http://ww2.sinaimg.cn/large/006tNc79ly1g3zf7di4bej30tu0jl75d.jpg)  
input name: input_2 with shape [1,46,46,38]  
 ![image](http://ww2.sinaimg.cn/large/006tNc79ly1g3zf7qwsgwj30ty0jktak.jpg)   
input name: input_3 with shape [1,46,46,19]  
 ![image](http://ww4.sinaimg.cn/large/006tNc79ly1g3zf7qfoxbj30tv0jstak.jpg)  
 output name: Mconv7_stage2_L1/BiasAdd  
 ![image](http://ww4.sinaimg.cn/large/006tNc79ly1g3zf7ql3gvj30ty0jpq53.jpg)  
 output name: Mconv8_stage2_L1/BiasAdd  
 ![image](http://ww2.sinaimg.cn/large/006tNc79ly1g3zf7qt2y5j30tw0jmabs.jpg)  
2. Check the network structure to make sure the model satisfying the "supported operation"  
2.1 Operations supported by Intel Movidius Neural Compute Stick  
2.2 [All operations supported by the Edge TPU and any known limitations](https://coral.withgoogle.com/docs/edgetpu/models-intro/)  
2.3 [Requirement of Edge TPU online compiler](https://coral.withgoogle.com/web-compiler/)

### Convert Tensorflow model to IR model using OpenVINO toolkit

```markdown
cd xxx@xxx:your_path/intel/openvino_xxxx.x.xxx/deployment_tools/model_optimizer
python3 mo_tf.py --input_model xxx.pb --input_shape [1,368,368,3],[1,46,46,38],[1,46,46,19] --input input_1,input_2,input_3 --data_type FP16 --output Mconv7_stage2_L1/BiasAdd,Mconv8_stage2_L1/BiasAdd --output_dir /tmp/
```

Note: the input_shape cannot set as -1 or None. Check the source code and graph on TensorBoard to make sure the input_shape and input parameters. 

### Run IR model on Intel Movidius Neural Compute Stick 

We use Pythn API for test. 

```markdown
cd your_path/model_test
python test_op.py
```

### Acceleration result
CPU (Intel core i5-4300U CPU) -- 234.9ms  
NCS (Intel Neutal compute stick - MA2540 VPU) -- 11.6ms


****

## Convert Keras model on Google Coral Edge TPU USB accelerator

### Convert Keras model to Tensorflow model

The same as before.

### Convert Tensorflow model to tensorflow lite model

Convert the tensorflow frozen graph (.pb) into tensorflow lite model (.lite) using tf.lite.TFLiteConverter.from_frozen_graph API:

```markdown
cd your_path/pb2lite
python convert.py
```

Note: 
1. check the graph on tensorboard for input_arrays and output_arrays
2. input_shapes: only need input_1 shape (1,368,368,3)

### Convert tensorflow lite model to tensorflow edge lite model

Google provide [Edge TPU Model Compiler](https://coral.withgoogle.com/web-compiler) for converting.  
Check the requirements and upload the tflite model, and the web complier convert the edge lite model automatically.

Note: if failed, according to the requirements, we need use TensorFlow Lite Optimizing Converter (TOCO) to get a fully-quantized TensorFlow Lite model:

```markdown
toco --input_file=op.pb --output_file=op_quantized.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=QUANTIZED_UINT8 --input_shape="1,368, 368,3" --input_array=input_1 --output_array="Mconv7_stage2_L1/BiasAdd","Mconv8_stage2_L1/BiasAdd" --std_value=128 --mean_value=128 --default_ranges_min=0 --default_ranges_max=6
```

### Run Edge TPU model (.tflite) on Google Coral Edge TPU USB accelerator


## Support or Contact

If any question, contact with [email](nicole2442@gmail.com) and we will help you sort it out.
