//
// Created by rrh4fe on 4/20/21.
//
# include "trt_dep.h"

#include <fstream>
#include <array>
#include <sstream>
#include <chrono>

Logger gLogger;

TrtSharedEnginePtr shared_engine_ptr(nvinfer1::ICudaEngine* ptr) {
    return TrtSharedEnginePtr (ptr, TrtDeleter());
}

TrtSharedEnginePtr parse_to_engine(std::string onnx_pth, bool use_fp16) {
    unsigned int maxBatchSize{1};
    int memory_limit = 1U << 30; //1G

    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if(!builder) {
        std::cout << "creating builder failed\n";
        std::abort();
    } else {
        std::cout << "Builder created\n";
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cout << "Network Creation failed \n";
        std::abort();
    }

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config){
        std::cout << "Builder config creation failed \n";
        std::abort();
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        std::cout << "Parser creation failed\n";
    }

    int verbosity = (int)nvinfer1::ILogger::Severity::kINFO;
    bool state = parser->parseFromFile(onnx_pth.c_str(), verbosity);

    if(!state) {
        std::cout << "Parsing of model failed\n";
        std::abort();
    }

    config->setMaxWorkspaceSize(memory_limit);
    if (use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto output = network->getOutput(0);
    output->setType(nvinfer1::DataType::kINT32);

    TrtSharedEnginePtr engine = shared_engine_ptr(
            builder->buildEngineWithConfig(*network, *config));
    if(!engine) {
        std::cout << "Engine Creation failed \n";
        std::abort();
    }

    return engine;
}

void serialize(TrtSharedEnginePtr engine, std::string save_path){
    auto trt_stream = TrtUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
    if (!trt_stream) {
        std::cout << "Engine serialization failed\n";
        std::abort();
    }

    std::ofstream ofile(save_path, std::ios::in | std::ios::binary);
    ofile.write((const char*)trt_stream->data(), trt_stream->size());

    ofile.close();
}

TrtSharedEnginePtr deserialize(std::string serpth){

    std::ifstream ifile(serpth, std::ios::in | std::ios::binary);
    if (!ifile){
        std::cout << "Reading of serialized file failed\n";
        std::abort();
    }

    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    std::cout<<"model size: "<< mdsize << std::endl;

    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    TrtSharedEnginePtr engine = shared_engine_ptr(
            runtime->deserializeCudaEngine((void *)&buf[0],mdsize, nullptr));
    return engine;
}

std::vector<int> infer_with_engine(TrtSharedEnginePtr engine, std::vector<float>& data) {
    nvinfer1::Dims3 out_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("preds")));

    const int batchsize{1}, H{out_dims.d[1]}, W{out_dims.d[2]};
    const int in_size{static_cast<int>(data.size())};
    const int out_size{batchsize * H * W};
    std::vector<void*> buffs(2);
    std::vector<int> res(out_size);

    auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context){
        std::cout << "Execution COntext creation failed\n";
        std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
        std::cout << "input memory allocation failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[1], out_size * sizeof (int));
    if (state){
        std::cout << "output memory allocation failed\n";
        std::abort();
    }

    cudaStream_t stream;
    state = cudaStreamCreate(&stream);
    if (state){
        std::cout << "Stream creation failed\n";
        std::abort();
    }

    state = cudaMemcpyAsync(
            buffs[0], &data[0], in_size * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    if (state){
        std::cout << "Transferring Input Tensor to GPU failed\n";
        std::abort();
    }
    context->enqueueV2(&buffs[0], stream, nullptr);
    state = cudaMemcpyAsync(
            &res[0], buffs[1], out_size* sizeof(int),
            cudaMemcpyDeviceToHost, stream
            );
    if(state)
    {
        std::cout << "Transferring Output Tensor to host failed\n";
        std::abort();
    }
    cudaStreamSynchronize(stream);

    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaStreamDestroy(stream);

    return res;
}

void test_fps_with_engine(TrtSharedEnginePtr engine)
{
    nvinfer1::Dims3 in_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    nvinfer1::Dims3 out_dims = static_cast<nvinfer1::Dims3&&>(
            engine->getBindingDimensions(engine->getBindingIndex("preds")));
    const int batchsize{1};
    const int oH{out_dims.d[1],}, oW{out_dims.d[2]};
    const int iH{in_dims.d[2]}, iW{in_dims.d[3]};
    const int in_size{batchsize * 3 * iH * iW};
    const int out_size{batchsize * oH * oW};

    auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if(!context){
        std::cout << "Execution context creation failed\n";
        std::abort();
    }

    std::vector<void*> buffs(2);
    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size* sizeof(float));
    if (state) {
        std::cout<<"Allocation of memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffs[1], out_size* sizeof(int));
    if (state) {
        std::cout<<"Allocation of memory failed\n";
        std::abort();
    }

    std::cout << "\n Testing with crop size of (" << iH <<","<<iW<<") ...\n";
    auto start = std::chrono::steady_clock::now();
    const int n_loops{1000};
    for (int i{0}; i<n_loops; ++i){
        context->executeV2(&buffs[0]);
    }

    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end-start).count();
    duration /= 1000;
    std::cout << "running" << n_loops <<"times, exec time: "<< duration << "s" << std::endl;
    std::cout << "fps is: " << static_cast <double>(n_loops)/duration << std::endl;

    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
}