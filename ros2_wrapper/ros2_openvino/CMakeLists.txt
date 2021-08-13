cmake_minimum_required(VERSION 3.5)
project(deployment_ovo)

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


find_library(OVO_INFERENCE_LIB ovoinference ${CMAKE_PREFIX_PATH}/lib/)
message(${OVO_INFERENCE_LIB})

find_path(OVO_INFERENCE_INCLUDES ovoinference.h ${CMAKE_PREFIX_PATH}/include/)

# find dependencies
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(inferwithovo src/inferwithovo.cpp)
target_include_directories(inferwithovo PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OVO_INFERENCE_INCLUDES}
        ${InferenceEngine_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )
target_link_libraries(inferwithovo
        ${OVO_INFERENCE_LIB}
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )
ament_target_dependencies(inferwithovo
        rclcpp
        image_transport
        OpenCV
        cv_bridge)

install(TARGETS
        inferwithovo
        DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
