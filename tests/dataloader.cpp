#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "dataloader.h"

TEST_CASE( "Test default image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.get(0, 0);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 256, 256} );
}

TEST_CASE( "Test unchanged image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.get(0, 0, -1);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 5, 5} );
}

TEST_CASE( "Test specific image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.get(0, 0, 42);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 42, 42} );
}

