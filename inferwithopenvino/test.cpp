//
// Created by rrh4fe on 7/8/21.
//

# include "ovoinference.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[] )
{
    try
    {
        if (argc != 5){
            std::cout << "Usage :" << argv[0] << "<path_to_model> <path_to_image> <device_name> <path_to_store_result>" << std::endl;
        }

        const std::string input_model {argv[1]};
        const std::string input_image_path {argv[2]};
        const std::string device_name = {argv[3]};
        const std::string save_pth = {argv[4]};

        auto infer = std::make_shared<ovoinference>(input_model, device_name);
        cv::Mat input_img = cv::imread(input_image_path);
        auto dims = infer->checkdims();
        cv::Mat result = infer->Segment(input_img);

        cv::imwrite(save_pth,result);
    }
    catch (const std::exception& ex){
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}