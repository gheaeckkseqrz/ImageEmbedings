#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "images.h"

TEST_CASE( "Load Red", "[Images]" )
{
  torch::Tensor red = loadImage("../tests/data/red.png");
  REQUIRE( red.sizes() == std::vector<int64_t>{3, 5, 5} );
  for (unsigned int y(0) ; y < 5 ; ++y)
    for (unsigned int x(0) ; x < 5 ; ++x)
      {
	REQUIRE( red[0][y][x].item().to<float>() == 0.0f );
	REQUIRE( red[1][y][x].item().to<float>() == 0.0f );
	REQUIRE( red[2][y][x].item().to<float>() == 1.0f );
      }
}

TEST_CASE( "Load Green", "[Images]" )
{
  torch::Tensor green = loadImage("../tests/data/green.png");
  REQUIRE( green.sizes() == std::vector<int64_t>{3, 5, 5} );
  for (unsigned int y(0) ; y < 5 ; ++y)
    for (unsigned int x(0) ; x < 5 ; ++x)
      {
	REQUIRE( green[0][y][x].item().to<float>() == 0.0f );
	REQUIRE( green[1][y][x].item().to<float>() == 1.0f );
	REQUIRE( green[2][y][x].item().to<float>() == 0.0f );
      }
}

TEST_CASE( "Load Blue", "[Images]" )
{
  torch::Tensor blue = loadImage("../tests/data/blue.png");
  REQUIRE( blue.sizes() == std::vector<int64_t>{3, 5, 5} );
  for (unsigned int y(0) ; y < 5 ; ++y)
    for (unsigned int x(0) ; x < 5 ; ++x)
      {
	REQUIRE( blue[0][y][x].item().to<float>() == 1.0f );
	REQUIRE( blue[1][y][x].item().to<float>() == 0.0f );
	REQUIRE( blue[2][y][x].item().to<float>() == 0.0f );
      }
}
