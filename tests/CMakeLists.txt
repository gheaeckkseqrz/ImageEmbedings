
add_executable(TestFeatureExtractor feature_extractor.cpp)
target_link_libraries(TestFeatureExtractor "${TORCH_LIBRARIES}")
add_test(TestFeatureExtractor TestFeatureExtractor)

add_executable(TestImages images.cpp ../images.cpp)
target_link_libraries(TestImages ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(TestImages TestImages)

add_executable(TestDataloader dataloader.cpp ../images.cpp)
target_link_libraries(TestDataloader ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(TestDataloader TestDataloader)
