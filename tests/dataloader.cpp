#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "dataloader.h"

TEST_CASE( "Test size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 3 );
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 6 );
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 9 );
}

TEST_CASE( "Test default image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.getImage(0, 0);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 256, 256} );
}

TEST_CASE( "Test unchanged image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.getImage(0, 0, -1);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 5, 5} );
}

TEST_CASE( "Test specific image size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  torch::Tensor t = d.getImage(0, 0, 42);
  REQUIRE( t.sizes() == std::vector<int64_t>{3, 42, 42} );
}

TEST_CASE( "Test limits", "[Dataloader]" )
{
  Dataloader d;
  for (int i(0) ; i < 50 ; ++i)
    d.addFolder("../tests/data");
  d.setLimits(2, 2);
  Triplet t = d.get(0);
  REQUIRE( t.anchor_folder_index < 2);
  REQUIRE( t.diff_folder_index < 2);
  REQUIRE( t.diff_folder_index != t.anchor_folder_index);
  REQUIRE( t.anchor_index < 2 );
  REQUIRE( t.same_index < 2 );
  REQUIRE( t.diff_index < 2 );
}

TEST_CASE( "Index to file folder", "[Dataloader]" )
{
  Dataloader d(12);
  d.addFolder("../tests/data");
  d.addFolder("../tests/data");
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 9u );

  // Folder 1
  REQUIRE( d.findFolderAndFileForIndex(0) == std::make_pair(0u, 0u) );
  REQUIRE( d.findFolderAndFileForIndex(1) == std::make_pair(0u, 1u) );
  REQUIRE( d.findFolderAndFileForIndex(2) == std::make_pair(0u, 2u) );

  // Folder 2
  REQUIRE( d.findFolderAndFileForIndex(3) == std::make_pair(1u, 0u) );
  REQUIRE( d.findFolderAndFileForIndex(4) == std::make_pair(1u, 1u) );
  REQUIRE( d.findFolderAndFileForIndex(5) == std::make_pair(1u, 2u) );

  // Folder 3
  REQUIRE( d.findFolderAndFileForIndex(6) == std::make_pair(2u, 0u) );
  REQUIRE( d.findFolderAndFileForIndex(7) == std::make_pair(2u, 1u) );
  REQUIRE( d.findFolderAndFileForIndex(8) == std::make_pair(2u, 2u) );
}
