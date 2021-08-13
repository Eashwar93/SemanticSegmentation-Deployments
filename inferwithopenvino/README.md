# Inference with Openvino
## A Inference library for Semantic Segmentation using Openvino

### Tested with:
Linux Ubuntu 18.04 (OSD 5)
Intel 9th gen CPU, GPU
Intel MYRIADX Neural Compute Stick 2

### Openvino Installation:
1. Please deactivate virtual environments as it interferes with the installation of Openvino python tools
2. Please refer to the installation [link](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)
3. Please configure the Intel GPU and NCS2 with openvino in the same link after installation
4. Always source the environment varialbles of openvino in every terminal using the command and it is recommended not to have it part of the .bashrc file unless you make use of openvino always
```bash   
source /opt/intel/openvino_2021/bin/setupvars.sh
```

### Model Optimizer:
#### Make sure to source the openvino enviroment variables
#### Model Conversion
Model optimizer is a tool from openvino that optimizes and converts the different models(ONNX models in our case) to IR models that is later used by openvino to perform inference at runtime
1. Use the script mo.sh to convert the ONNX models to IR models
2. Modify the arguments -m  and -o accordingly to set the paths correctly
3. The other arguments are matched in accordance with the parameters in the training pipeline so modify them with caution if needed
4. This coverter generates 3 files, the openvino must be directed to the .xml file in the future to perform inference

### Benchmark App:
#### Make sure to source the openvino enviroment variables
The openvino comes with a python application that lets you test the FPS on a target hardware
Use the following script becnchmark.sh to test the FPS of your model on a target hardware. Make sure to modify the -d argument to point to the right hardware of your interest


### Clone and build the repo:
#### Make sure to source the openvino enviroment variables
```bash
git clone ssh://git@sourcecode.socialcoding.bosch.com:7999/rcl/rcl-3d-perception.git
git checkout thesis/deployment_pipeline
cd inferwithovo
mkdir build && cd build
cmake ..
sudo make install
```

This installs the include and library in a central location in your computer usually at 
```bash   
/usr/local/include
/usr/local/lib
```
### Segmentation Demo:
#### Make sure to source the openvino enviroment variables
Inside the build directory there will also be a executable "test". 

Use the following command to perform a semantic segmentation:
```bash
./test </path/to/.xml/file> </path/to/image> <path/to/segmentedresult>
```

