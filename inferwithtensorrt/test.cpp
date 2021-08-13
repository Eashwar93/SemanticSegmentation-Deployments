//
// Created by rrh4fe on 5/8/21.
//
#include "trtinference.h"
#include "string"
#include <opencv2/opencv.hpp>


int main()
{
    std::string model_pth = "/home/rrh4fe/deployment_pipeline/inferwithtrt/example/monorail_model_fp16.trt";
    std::string image_path = "/home/rrh4fe/deployment_pipeline/inferwithtrt/example/1.png";
    std::string pred_path ="/home/rrh4fe/deployment_pipeline/inferwithtrt/example/1_res.png";
    cv::Mat input_img = cv::imread(image_path);

    auto infer = std::make_shared<trtinference>(model_pth);
    std::vector<int> dims = infer->checkdims();
    cv::Mat result = infer->Segment(input_img);

    cv::imwrite(pred_path, result);

}