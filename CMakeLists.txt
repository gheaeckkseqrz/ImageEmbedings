cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
project(ImageEmbeddings)

find_package(OpenCV REQUIRED)
set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch  REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(. ${OpenCV_INCLUDE_DIRS})

add_executable(ImageEmbeddings main.cpp images.cpp tsne.cpp sptree.cpp)
target_link_libraries(ImageEmbeddings ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

enable_testing()
add_subdirectory(tests)
