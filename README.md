# Welcome to hardware acceleration page

The model we convert is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) algorithm. We accelerate this algorithm from two sides: software and hardware. For the software acceleration part, we simplify the network structure for acceleration and achieve it by Keras. 

After that, we try hardware acceleration. We choose two platforms: [Intel Movidius Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick) and [Google Coral Edge TPU USB accelerator](https://coral.withgoogle.com).

## Convert Keras model on Intel Movidius Neural Compute Stick 

There are two ways for transitioning provided by Intel: OpenVINO toolkit and NCSDK. In this blog, we use OpenVINO toolkit. If you want to know how to use NCSDK to convert model and the difference between OpenVINO toolkit and NCSDK, more details are [here](https://software.intel.com/en-us/articles/transitioning-from-intel-movidius-neural-compute-sdk-to-openvino-toolkit).

### OpenVINO installation and test

Step 1. Follow the [tutorial](developer.movidius.com/start) to install OpenVINO toolkit. 

After installation, there are basic three folders in our home dictionary:

xxx@xxx:intel/openvino_xxxx.x.xxx
xxx@xxx:inference_engine_samples_build
xxx@xxx:openvino_models

Step 2. Run demo

```markdown
cd intel/openvino_xxxx.x.xxx/deployment_tools/demo/demo_squeezenet_download_convert_run.sh
```

Note: the "target" and "target_precision" need preset before run .sh:
if use CPU: tartget="CPU" and target_precision="FP32"
if use Neural Compute Stick: target="MYRIAD" and target_precision="FP16"

### convert Keras model to Tensorflow model

Step 1. train and saving the Keras model

Keras model is the file .h5. Because the nerwork structure and weights are all need for model transitioning, we save model with setting the parameter save_weights_only as false.

```markdown
checkpoint = ModelCheckPoint(weights_best_file, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='min', period=1)
```

Step 2. convert Keras model to Tensorflow model

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

After that, we convert Keras model (.h5) to tensowflow model (.pb). The tensorfow model is the frozen graph.

### convert Tensorflow model to IR model using OpenVINO toolkit

```markdown
cd xxx@xxx:intel/openvino_xxxx.x.xxx/deployment_tools/model_optimizer
python3 mo_tf.py --input_model xxx.pb --input_shape [1,368,368,3],[1,46,46,38],[1,46,46,19] --input input_1,input_2,input_3 --data_type FP16 --output_dir /tmp/
```

Note: the input_shape cannot set as -1 or None. Check the source code and graph on tensorboard to make sure the input_shape and input parameters. 

### run IR model on Intel Movidius Neural Compute Stick 

We use Pythn API for test. 

## Convert Keras model on Google Coral Edge TPU USB accelerator

### convert Keras model to Tensorflow model

The same as before.

### convert Tensorflow model to tensorflow lite model

Use tf.lite.TFLiteConverter.from_frozen_graph API

Note: 
1. check the graph on tensorboard for input_arrays and output_arrays
2. input_shapes: only need input_1 shape (1,368,368,3)

### convert tensorflow lite model to tensorflow edge lite model

Google provide [Edge TPU Model Compiler](https://coral.withgoogle.com/web-compiler) for converting.

Check the requirements and upload the tflite model, and the web complier convert the edge lite model automatically.

### run Edge TPU model (.tflite) on Google Coral Edge TPU USB accelerator



## Support or Contact

If any question, contact with [email](nicolekliao@163.com) and we will help you sort it out.
