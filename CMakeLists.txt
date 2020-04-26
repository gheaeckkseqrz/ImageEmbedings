cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(Faces)

find_package(OpenCV REQUIRED)
find_package(Torch  REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(. ${OpenCV_INCLUDE_DIRS})
add_executable(Faces main.cpp images.cpp)
target_link_libraries(Faces ${TORCH_LIBRARIES} ${OpenCV_LIBS})

enable_testing()
add_subdirectory(tests)
