
add_executable(TestFeatureExtractor feature_extractor.cpp)
target_link_libraries(TestFeatureExtractor ${TORCH_LIBRARIES})
add_test(NAME TestFeatureExtractor COMMAND TestFeatureExtractor WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TestImages images.cpp ../images.cpp)
target_link_libraries(TestImages ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(NAME TestImages COMMAND TestImages WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TestDataloader dataloader.cpp ../images.cpp)
target_link_libraries(TestDataloader ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
add_test(NAME TestDataloader COMMAND TestDataloader WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TestDataminer dataminer.cpp ../images.cpp)
target_link_libraries(TestDataminer ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
add_test(NAME TestDataminer COMMAND TestDataminer WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
