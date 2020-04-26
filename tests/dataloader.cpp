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

TEST_CASE( "Test limits", "[Dataloader]" )
{
  Dataloader d;
  for (int i(0) ; i < 50 ; ++i)
    d.addFolder("../tests/data");
  d.setLimits(2, 2);
  Triplet t = d.getTriplet();
  REQUIRE( t.anchor_folder_index < 2);
  REQUIRE( t.diff_folder_index < 2);
  REQUIRE( t.diff_folder_index != t.anchor_folder_index);
  REQUIRE( t.anchor_index < 2 );
  REQUIRE( t.same_index < 2 );
  REQUIRE( t.diff_index < 2 );
}

