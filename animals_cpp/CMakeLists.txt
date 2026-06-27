cmake_minimum_required(VERSION 3.18)
project(animals_classifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "E:/libtorch")

find_package(Torch REQUIRED)

# OpenCV
set(OpenCV_DIR "E:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

add_executable(predict predict.cpp)

target_link_libraries(predict 
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS}
)

# Auto copy LibTorch DLLs
file(GLOB TORCH_DLLS "E:/libtorch/lib/*.dll")
add_custom_command(TARGET predict POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:predict>)

# Auto copy OpenCV DLL
add_custom_command(TARGET predict POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "E:/opencv/build/x64/vc16/bin/opencv_world4120.dll"
    $<TARGET_FILE_DIR:predict>)