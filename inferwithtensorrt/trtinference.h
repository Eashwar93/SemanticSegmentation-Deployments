//
// Created by rrh4fe on 5/8/21.
//

#ifndef INFERWITHTRT_TRTINFERENCE_H
#define INFERWITHTRT_TRTINFERENCE_H

#include "trt_dep.h"
#include <opencv2/opencv.hpp>

class trtinference {
public:

    trtinference(std::string &model_path);
    std::vector<int> checkdims();
    cv::Mat Segment(cv::Mat &input_img);


private:
    TrtSharedEnginePtr engine;
    nvinfer1::Dims3 i_dims;
    nvinfer1::Dims3 o_dims;
    int iH, iW, oH, oW;
    std::array<float,3> mean;
    std::array<float,3> variance;
    float scale;
    std::vector<float> data;
    std::vector<std::vector<uint8_t>> color_map;

    std::vector<std::vector<uint8_t>> get_color_map();


};


#endif //INFERWITHTRT_TRTINFERENCE_H
