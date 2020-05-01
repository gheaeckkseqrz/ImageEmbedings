cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(Faces)

find_package(OpenCV REQUIRED)
find_package(Torch  REQUIRED)
find_package(SFML COMPONENTS graphics REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(. ${OpenCV_INCLUDE_DIRS})
add_executable(Faces main.cpp images.cpp)
target_link_libraries(Faces ${TORCH_LIBRARIES} ${OpenCV_LIBS} sfml-graphics Threads::Threads)

message( "SFML ${SFML_LIBS}" )

find_package (Threads)
add_executable(GUI GUI.cpp)
target_link_libraries(GUI sfml-graphics Threads::Threads)

enable_testing()
add_subdirectory(tests)
