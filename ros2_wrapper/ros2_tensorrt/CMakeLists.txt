cmake_minimum_required(VERSION 3.5)
project(deployment_trt)

link_directories(/usr/local/cuda/lib64)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


LIST(APPEND CMAKE_PREFIX_PATH "/usr/local")
message(${CMAKE_PREFIX_PATH}/lib)

find_library(TRT_INFERENCE_LIB trtinference ${CMAKE_PREFIX_PATH}/lib/)
message(${TRT_INFERENCE_LIB})
find_path(TRT_INFERENCE_INCLUDES trtinference.h ${CMAKE_PREFIX_PATH}/include/)

find_library(TRT_DEP_LIB trt_dep /usr/local/lib/)
message(${TRT_DEP_LIB})
find_path(TRT_DEP_INCLUDES trt_dep.h /usr/local/include/)


# find dependencies
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(inferwithtrt src/inferwithtrt.cpp)
target_include_directories(inferwithtrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include> ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inferwithtrt ${TRT_INFERENCE_LIB} ${TRT_DEP_LIB} ${CUDA_LIBRARIES})
ament_target_dependencies(inferwithtrt rclcpp image_transport OpenCV cv_bridge )

install(TARGETS
        inferwithtrt
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS
        inferwithtrt
        DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
