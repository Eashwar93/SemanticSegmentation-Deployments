cmake_minimum_required(VERSION 3.17)
project(inferwithovo)

set(CMAKE_CXX_STANDARD 14)

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)



add_library(ovoinference SHARED ovoinference.cpp ovoinference.h)
target_include_directories(ovoinference PUBLIC
        ${InferenceEngine_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ovoinference
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES})


add_executable(test test.cpp)
target_include_directories(test PUBLIC
        ${InferenceEngine_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ovoinference)

install(TARGETS ovoinference LIBRARY DESTINATION lib)
install(FILES ovoinference.h DESTINATION include)