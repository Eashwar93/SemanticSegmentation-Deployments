//
// Created by rrh4fe on 7/8/21.
//

#ifndef INFERWITHOVO_OVOINFERENCE_H
#define INFERWITHOVO_OVOINFERENCE_H

#include <iostream>
#include <vector>
#include "inference_engine.hpp"
#include <opencv2/opencv.hpp>

class ovoinference {
public:
    ovoinference(const std::string &model_path, const std::string &device_name);
    cv::Mat Segment(cv::Mat &input_img);
    std::vector<size_t> checkdims();

private:
    std::vector<std::vector<uint8_t>> get_color_map();
    InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat);



    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InputInfo::Ptr input_info;
    std::string input_name;
    InferenceEngine::DataPtr output_info;
    std::string output_name;

    InferenceEngine::InferRequest infer_request;

    size_t iH, iW, oH, oW;
};


#endif //INFERWITHOVO_OVOINFERENCE_H
