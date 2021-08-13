#!/bin/bash

python3 /opt/intel/openvino_2021/deployment_tools/tools/benchmark_tool/benchmark_app.py -m ~/openvino_models/ir/bisenetv1_g1.xml -d MYRIAD -niter 500
