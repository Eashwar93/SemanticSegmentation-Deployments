cmake_minimum_required(VERSION 3.10)
project(inferwithtrt)

set(CMAKE_CXX_FLAGS "-std=c++14 -O1")

link_directories(/usr/local/cuda/lib64)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_library(trtinference SHARED trtinference.cpp trtinference.h trt_dep.cpp)
target_include_directories(
        trtinference PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(trtinference -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES})

add_library(trt_dep SHARED trt_dep.cpp trt_dep.h)
target_include_directories(
        trt_dep PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(trtinference -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES})

add_executable(prepinferwithtrt prepinferwithtrt.cpp trt_dep.cpp)
target_include_directories(
        prepinferwithtrt PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
        prepinferwithtrt -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

add_executable(testmain test.cpp trtinference.cpp trt_dep.cpp)
target_include_directories(
        testmain PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
        testmain -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser trtinference
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

install(TARGETS trtinference LIBRARY DESTINATION lib)
install(FILES trtinference.h DESTINATION include)

install(TARGETS trt_dep LIBRARY DESTINATION lib)
install(FILES trt_dep.h DESTINATION include)