#!/bin/bash

python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py -m ~/openvino_models/onnx/bisenetv1_g1.onnx -o ~/openvino_models/ir --input_shape [1,3,480,640] --data_type FP16 --output preds --input input_image --mean_values [123.675,116.28,103.52] --scale_values [58.395,57.12,57.375]
