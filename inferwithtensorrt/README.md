# Inference with TensorRT
## A Inference library for Semantic Segmentation using TensorRT

### Tested with:
1. Linux Ubuntu 18.04 (OSD 5) on Intel 9th gen CPU, Nvidia Quadro T2000
2. JetPack 4.5.1 on Nvidia Jetson Nano, Xavier NX

### TensorRT Installation:
#### Linux Machine
1. Install Nvidia properaitary driver if you have installed 3rd party driver which usually comes pre-installed in OSD5 (Library tested with driver version 450.51.05)
2. Install suitable CUDA version from the [official page](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (Library tested with CUDA version 11.0)
3. Install suitable CUDnn version from the [official page](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) (Library tested with CUDNN version 8.0.5)
4. Install suitable TensorRT version from the [official page](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) (Library tested with TensorRT 7.2.3)
#### Jetson Computers
1. Flash the SD card with Jetpack 4.5.1 from the [official page](https://developer.nvidia.com/embedded/jetpack)

### Dependencies
1. OpenCV

### Clone the and build the repo:

```bash
git clone ssh://git@sourcecode.socialcoding.bosch.com:7999/rcl/rcl-3d-perception.git
git checkout thesis/deployment_pipeline
cd inferwithtrt
mkdir build && cd build
cmake ..
sudo make install
```

This installs the include and library in a central location in your computer usually at 
```bash   
/usr/local/include
/usr/local/lib
```
### Convert the model:

Inside the build directory there will a executable "prepinferwithtrt". 

Use the following command to compile the model for the specific hardware target
```bash
 ./prepinferwithtrt compile <path/to/onnx/model> <path/to/trt/model> --fp16
```

### Test the model speed
Use the following command to test FPS of the model for the specific hardware target
```bash
 ./prepinferwithtrt test <path/to/trt/model>
```

### Segmentation Demo
Use the following command to test FPS of the model for the specific hardware target
```bash
 ./prepinferwithtrt run <path/to/trt/model> <path/to/input/image> <path/to/segmented/image>
```
