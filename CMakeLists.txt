cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Faces)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Faces main.cpp)
target_link_libraries(Faces "${TORCH_LIBRARIES}")
set_property(TARGET Faces PROPERTY CXX_STANDARD 14)
