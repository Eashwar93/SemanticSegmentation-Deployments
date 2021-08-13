//
// Created by rrh4fe on 7/8/21.
//

#include "ovoinference.h"
#include <random>

ovoinference::ovoinference(const std::string &model_path,const std::string &device_name) {

    network = core.ReadNetwork(model_path);

    if (network.getOutputsInfo().size() != 1)
        throw std::logic_error("Inference Engine supports only single frame inference output");

    if (network.getInputsInfo().size() != 1)
        throw std::logic_error("Inference Engine supports only single input");

    input_info.swap(network.getInputsInfo().begin()->second);
    input_info->setPrecision(InferenceEngine::Precision::U8);
    input_info->setLayout(InferenceEngine::Layout::NCHW);
    input_info->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
    input_name = network.getInputsInfo().begin()->first;

    output_info.swap(network.getOutputsInfo().begin()->second);
    output_info->setPrecision(InferenceEngine::Precision::I32);
    output_info->setLayout(InferenceEngine::Layout::CHW);
    output_name = network.getOutputsInfo().begin()->first;

    executable_network = core.LoadNetwork(network, device_name);
    infer_request = executable_network.CreateInferRequest();

    iH = input_info->getTensorDesc().getDims()[2];
    iW = input_info->getTensorDesc().getDims()[3];
    oH = output_info->getTensorDesc().getDims()[1];
    oW = output_info->getTensorDesc().getDims()[2];

}

std::vector<std::vector<uint8_t>> ovoinference::get_color_map()
{
    std::vector<std::vector<uint8_t>> color_map(256, std::vector<uint8_t>(3));
    std::minstd_rand rand_engg(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (int i{0}; i < 256; ++i) {
        for (int j{0}; j < 3; j++) {
            color_map[i][j] = u(rand_engg);
        }
    }
    return color_map;
}

InferenceEngine::Blob::Ptr ovoinference::wrapMat2Blob(const cv::Mat &mat)
{
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
            strideW == channels &&
            strideH == channels * width;

    if (!is_dense)
        throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

cv::Mat ovoinference::Segment(cv::Mat &input_img)
{
    InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(input_img);
    imgBlob->allocate();

    infer_request.SetBlob(input_name, imgBlob);
    infer_request.Infer();
    imgBlob->deallocate();
    InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);


    auto const memLocker = output->cbuffer();
    const auto *res = memLocker.as<const int *>();


    cv::Mat pred(cv::Size(oW, oH), CV_8UC3);
    std::vector<std::vector<uint8_t>> color_map = get_color_map();

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

std::vector<size_t> ovoinference::checkdims()
{
    std::vector<size_t> dims = {iH, iW, oH, oW};
    return dims;
}

