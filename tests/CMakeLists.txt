
add_executable(TestFeatureExtractor feature_extractor.cpp)
target_link_libraries(TestFeatureExtractor "${TORCH_LIBRARIES}")
add_test(TestFeatureExtractor TestFeatureExtractor WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TestImages images.cpp ../images.cpp)
target_link_libraries(TestImages ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(TestImages TestImages WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TestDataloader dataloader.cpp ../images.cpp)
target_link_libraries(TestDataloader ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(TestDataloader TestDataloader WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
