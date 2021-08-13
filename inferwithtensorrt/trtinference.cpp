//
// Created by rrh4fe on 5/8/21.
//

#include "trtinference.h"
#include <random>

trtinference::trtinference(std::string &model_path)
{
    engine = deserialize(model_path);
    i_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("input_image"))
            );
    o_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("preds"))
            );
    iH = i_dims.d[2];
    iW = i_dims.d[3];
    oH = o_dims.d[1];
    oW = o_dims.d[2];

    mean={0.485f,0.456f,0.406f};
    variance={0.229f,0.224f,0.225f};
    scale = 1.f/ 255.f;
    for (int i{0}; i<3; ++i) {
        variance[i] = 1.f / variance[i];
    }
    data.resize(iH * iW * 3);
    color_map = get_color_map();
}

cv::Mat trtinference::Segment(cv::Mat &input_img)
{
    for (int h{0}; h<iH; ++h)
    {
        auto *p = input_img.ptr<cv::Vec3b>(h);
        for (int w{0}; w<iW; ++w)
        {
            for (int c{0}; c<3; ++c)
            {
                int idx = (2-c) * iH * iW + h * iW + w;
                data[idx] = (p[w][c]*scale -mean[c]) * variance[c];
            }
        }
    }
    std::vector<int> res = infer_with_engine(engine, data);
    cv::Mat pred(cv::Size(oW, oH), CV_8UC3);
    int idx{0};
    for (int i = 0; i < oH; ++i)
    {
        auto *ptr = pred.ptr<uint8_t>(i);
        for (int j = 0; j < oW; ++j)
        {
            ptr[0] = color_map[res[idx]][0];
            ptr[1] = color_map[res[idx]][1];
            ptr[2] = color_map[res[idx]][2];
            ptr += 3;
            ++idx;
        }
    }
    return pred;
}

std::vector<int> trtinference::checkdims()
{
    std::vector<int> dims = {iH, iW, oH, oW};
    return dims;
}

std::vector<std::vector<uint8_t>> trtinference::get_color_map()
{
    std::vector<std::vector<uint8_t>> color_map (256, std::vector<uint8_t>(3));
    std::minstd_rand rand_engg(123);
    std::uniform_int_distribution<uint8_t> u(0,255);
    for (int i{0}; i<256; ++i)
    {
        for (int j{0}; j<3; j++)
        {
            color_map[i][j] = u(rand_engg);
        }
    }
    return color_map;
}
