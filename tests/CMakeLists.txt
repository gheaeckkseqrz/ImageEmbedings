
add_executable(TestFeatureExtractor feature_extractor.cpp)
target_link_libraries(TestFeatureExtractor "${TORCH_LIBRARIES}")
add_test(TestFeatureExtractor TestFeatureExtractor)
