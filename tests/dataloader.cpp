#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "dataloader.h"

TEST_CASE( "Test size", "[Dataloader]" )
{
  Dataloader d;
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 1 );
  REQUIRE( d.nbIdentities() == 1 );
  REQUIRE( d.nbImages() == 3 );
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 2 );
  REQUIRE( d.nbIdentities() == 2 );
  REQUIRE( d.nbImages() == 6 );
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 3 );
  REQUIRE( d.nbIdentities() == 3 );
  REQUIRE( d.nbImages() == 9 );
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
  REQUIRE( t.anchor_folder_index[0].item<int64_t>() < 2);
  REQUIRE( t.diff_folder_index[0].item<int64_t>() < 2);
  REQUIRE( t.diff_folder_index[0].item<int64_t>() != t.anchor_folder_index[0].item<int64_t>());
  REQUIRE( t.anchor_index[0].item<int64_t>() < 2 );
  REQUIRE( t.same_index[0].item<int64_t>() < 2 );
  REQUIRE( t.diff_index[0].item<int64_t>() < 2 );
}

TEST_CASE( "Index to file folder", "[Dataloader]" )
{
  Dataloader d(12);
  d.addFolder("../tests/data");
  d.addFolder("../tests/data");
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 3u );
  REQUIRE( d.nbIdentities() == 3 );
  REQUIRE( d.nbImages() == 9 );


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

TEST_CASE( "Use with dataloader", "[Dataloader]")
{
  Dataloader d(12);
  for (unsigned int i(0) ; i < 9 ; ++i)
    d.addFolder("../tests/data");
  REQUIRE( *d.size() == 9u );
  REQUIRE( d.nbIdentities() == 9 );
  REQUIRE( d.nbImages() == 27 );

  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(d, 4);
  auto batch = data_loader->begin();

  REQUIRE( batch->anchor.sizes() == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->diff.sizes()   == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->same.sizes()   == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->anchor_folder_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->diff_folder_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->anchor_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->same_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->diff_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->anchor_folder_index[0].item<int64_t>() == 0 );
  REQUIRE( batch->anchor_folder_index[1].item<int64_t>() == 1 );
  REQUIRE( batch->anchor_folder_index[2].item<int64_t>() == 2 );
  REQUIRE( batch->anchor_folder_index[3].item<int64_t>() == 3 );
  ++batch;
  REQUIRE( batch->anchor.sizes() == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->diff.sizes()   == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->same.sizes()   == std::vector<int64_t>{4, 3, 12, 12} );
  REQUIRE( batch->anchor_folder_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->diff_folder_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->same_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->diff_index.sizes() == std::vector<int64_t>{4} );
  REQUIRE( batch->anchor_folder_index[0].item<int64_t>() == 4 );
  REQUIRE( batch->anchor_folder_index[1].item<int64_t>() == 5 );
  REQUIRE( batch->anchor_folder_index[2].item<int64_t>() == 6 );
  REQUIRE( batch->anchor_folder_index[3].item<int64_t>() == 7 );
  ++batch;
  REQUIRE( batch->anchor.sizes() == std::vector<int64_t>{1, 3, 12, 12} );
  REQUIRE( batch->diff.sizes()   == std::vector<int64_t>{1, 3, 12, 12} );
  REQUIRE( batch->same.sizes()   == std::vector<int64_t>{1, 3, 12, 12} );
  REQUIRE( batch->anchor_folder_index.sizes() == std::vector<int64_t>{1} );
  REQUIRE( batch->diff_folder_index.sizes() == std::vector<int64_t>{1} );
  REQUIRE( batch->same_index.sizes() == std::vector<int64_t>{1} );
  REQUIRE( batch->diff_index.sizes() == std::vector<int64_t>{1} );
  REQUIRE( batch->anchor_folder_index[0].item<int64_t>() == 8 );
  ++batch;
  REQUIRE( batch == data_loader->end() );
}

TEST_CASE( "Test filtering", "[Dataloader]")
{
  Dataloader d(256, "red.png");
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 1 );
  REQUIRE( d.nbIdentities() == 1 );
  REQUIRE( d.nbImages() == 1 );
}
