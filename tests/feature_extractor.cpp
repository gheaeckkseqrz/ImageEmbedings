#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "feature_extractor.h"

TEST_CASE( "OutputSize", "[FeatureExtractor]" )
{
  FeatureExtractor fe(1, 12);
  {
    torch::Tensor input = torch::rand({1, 3, 256, 256});
    torch::Tensor output = fe->forward(input);
    REQUIRE( output.sizes() == std::vector<int64_t>{1, 12} );
  }
  {
    torch::Tensor input = torch::rand({8, 3, 256, 256});
    torch::Tensor output = fe->forward(input);
    REQUIRE( output.sizes() == std::vector<int64_t>{8, 12} );
  }
}
