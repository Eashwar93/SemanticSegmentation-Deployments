//
// Created by rrh4fe on 4/20/21.
//

#ifndef INFERWITHTRT_TRT_DEP_H
#define INFERWITHTRT_TRT_DEP_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "NvInferRuntimeCommon.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>

class Logger: public nvinfer1::ILogger{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override{
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

struct TrtDeleter{
    template<typename T>
    void operator()(T* obj) const {   //need to understand this syntax
        if (obj)
        {obj->destroy();}
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;
using TrtSharedEnginePtr = std::shared_ptr<nvinfer1::ICudaEngine>;

extern Logger gLogger;

TrtSharedEnginePtr shared_engine_ptr(nvinfer1::ICudaEngine* ptr);
TrtSharedEnginePtr parse_to_engine(std::string onnx_path, bool use_fp16);
void serialize(TrtSharedEnginePtr engine, std::string save_path);
TrtSharedEnginePtr deserialize(std::string serpth);
std::vector<int> infer_with_engine(TrtSharedEnginePtr engine, std::vector<float>& data);
void test_fps_with_engine(TrtSharedEnginePtr engine);



#endif //INFERWITHTRT_TRT_DEP_H
