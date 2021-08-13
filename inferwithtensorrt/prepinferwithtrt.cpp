#include "trt_dep.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <array>
#include <sstream>
#include <random>


std::vector<std::vector<uint8_t>> get_color_map();

void compile_onnx(std::vector<std::string> args);
void run_with_trt(std::vector<std::string> args);
void test_speed(std::vector<std::string> args);

int main(int argc, char*argv[]){
    std::vector<std::string> args;
    for (int i{1}; i < argc; ++i) args.emplace_back(argv[i]);
    if (argc < 3){
        std::cout << "Minimum necessary command line arguments is 3\n";
    }

    if (args[0] == "compile") {
        if (argc < 4){
            std::cout << "Usage is ./inferwithtrt compile /path/to/onnxmodel /path/to/trtmodel [--fp16]\n";
            std::abort();
        }
        compile_onnx(args);
    }

    if (args[0] == "run") {
        if (argc < 5) {
            std::cout << "Usage is ./inferwithtrt run /path/to/trtmodel /path/to/inputimage /path/to/outputimage\n";
            std::abort();
        }
        run_with_trt(args);
    }

    if (args[0] == "test"){
        if (argc < 3) {
            std::cout << "Usgae is ./inferwithtrt test /path/to/trtmodel\n";
            std::abort();
        }
        test_speed(args);
    }

    return 0;
}

void compile_onnx(std::vector<std::string> args){
    bool use_fp16{false};
    if((args.size() >= 4 && args[3] == "--fp16")) use_fp16 = true;

    TrtSharedEnginePtr engine = parse_to_engine(args[1], use_fp16);
    serialize(engine, args[2]);
}

void run_with_trt(std::vector<std::string> args){
    TrtSharedEnginePtr engine = deserialize(args[1]);

    nvinfer1::Dims3 i_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    nvinfer1::Dims3 o_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("preds")));
    const int iH{i_dims.d[2]}, iW{i_dims.d[3]};
    const int oH{o_dims.d[1]}, oW{o_dims.d[2]};

    cv::Mat im = cv::imread(args[2]);
    if (im.empty()){
        std::cout << "cannot read image \n";
        std::abort();
    }

    int orgH{im.rows}, orgW(im.cols);
    if ((orgH != iH) || (orgW != iW))
    {
        std::cout << "Image dimension do not match the necessary input size to the image\n";
        cv::resize(im, im, cv::Size(iW, iH), cv::INTER_CUBIC);
    }

    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> variance{0.229f, 0.224f, 0.225f};
    float scale = 1.f / 255.f;
    for (int i{0}; i<3; ++i) {
        variance[i] = 1.f / variance[i];
    }
    std::vector<float> data(iH * iW * 3);
    for (int h{0}; h<iH; ++h)
    {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w<iW; ++w)
        {
           for (int c{0}; c<3; ++c)
           {
               int idx = (2 - c) * iH * iW + h * iW + w;
               data[idx] = (p[w][c] * scale -mean[c]) * variance[c];
           }
        }
    }

    std::vector<int> res = infer_with_engine(engine, data);

    std::vector<std::vector<uint8_t>> color_map = get_color_map();
    cv::Mat pred(cv::Size(oW, oH), CV_8UC3);
    int idx{0};
    for (int i{0}; i < oH; ++i)
    {
        uint8_t *ptr = pred.ptr<uint8_t>(i);
        for (int j{0}; j < oW; ++j)
        {
            ptr[0] = color_map[res[idx]][0];
            ptr[1] = color_map[res[idx]][1];
            ptr[2] = color_map[res[idx]][2];
            ptr += 3;
            ++ idx;
        }
    }

    if ((orgH != oH) || (orgW != oW)) {
        cv::resize(pred, pred, cv::Size(orgW, orgH), cv::INTER_NEAREST);
    }
    cv::imwrite(args[3], pred);

}

std::vector<std::vector<uint8_t>> get_color_map(){
    std::vector<std::vector<uint8_t>> color_map(256, std::vector<uint8_t>(3));
    std::minstd_rand rand_engg(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (int i{0}; i < 256; ++i) {
        for (int j{0}; j<3; ++j) {
            color_map[i][j] = u(rand_engg);
        }
    }
    return color_map;
}

void test_speed(std::vector<std::string> args){
    TrtSharedEnginePtr engine = deserialize(args[1]);
    test_fps_with_engine(engine);
}