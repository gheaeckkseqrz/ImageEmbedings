cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
project(ImageEmbeddings)

find_package(SFML COMPONENTS graphics REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch  REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(. ${OpenCV_INCLUDE_DIRS})

add_executable(TrainEmbeddings train.cpp images.cpp)
target_link_libraries(TrainEmbeddings ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

add_executable(VisualiseEmbeddings visualise.cpp images.cpp tsne.cpp sptree.cpp)
target_link_libraries(VisualiseEmbeddings ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs sfml-graphics Threads::Threads)

enable_testing()
add_subdirectory(tests)
