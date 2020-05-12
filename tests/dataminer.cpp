#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "dataminer.h"

TEST_CASE( "Initial state", "[Dataminer]" )
{
  Dataminer d(12);
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 3 );
  torch::Tensor embedding = d.getEmbedding(1);
  REQUIRE( embedding.sizes() == std::vector<int64_t>({12}) );
  for (unsigned int i(0) ; i < 12 ; ++i)
    REQUIRE (embedding[i].item<float>() == 0.0f );
}

TEST_CASE( "Set/Get embedding", "[Dataminer]" )
{
  unsigned int Z = 4;
  Dataminer d(Z);

  // Add 30 items
  for (unsigned int i(0) ; i < 10 ; ++i)
    d.addFolder("../tests/data");
  REQUIRE( *d.size() == 30 );

  // Check everything is initialized to 0
  for (unsigned int i(0) ; i < 30 ; ++i)
    {
      torch::Tensor embedding = d.getEmbedding(i);
      REQUIRE( embedding.sizes() == std::vector<int64_t>({Z}) );
      for (unsigned int j(0) ; j < Z ; ++j)
  	REQUIRE (embedding[j].item<float>() == 0.0f );
    }

  // Create embedding
  torch::Tensor t = torch::zeros({Z});
  float v[] = {1.0f, 2.0f, 3.0f, 4.0f};
  for (unsigned int i(0) ; i < Z ; ++i)
    t[i] = v[i];

  // Set The embedding
  d.setEmbedding(12, t);

  // Check new values
  for (unsigned int i(0) ; i < 30 ; ++i)
    {
      torch::Tensor embedding = d.getEmbedding(i);
      REQUIRE( embedding.sizes() == std::vector<int64_t>({Z}) );
      if (i == 12)
  	{
  	  REQUIRE (embedding[0].item<float>() == 1.0f );
  	  REQUIRE (embedding[1].item<float>() == 2.0f );
  	  REQUIRE (embedding[2].item<float>() == 3.0f );
  	  REQUIRE (embedding[3].item<float>() == 4.0f );
  	}
      else
  	{
  	  for (unsigned int j(0) ; j < Z ; ++j)
  	    REQUIRE (embedding[j].item<float>() == 0.0f );
  	}
    }
}

TEST_CASE( "Find Closest", "[Dataminer]" )
{
  unsigned int Z = 2;
  Dataminer d(Z);

  // Add 6 items
  d.addFolder("../tests/data");
  d.addFolder("../tests/data");
  REQUIRE( *d.size() == 6 );

  // Set embeddings folder 1
  torch::Tensor t1 = torch::zeros({Z});
  t1[0] =  -1;
  t1[1] =   1;
  torch::Tensor t2 = torch::zeros({Z});
  t2[0] =   0;
  t2[1] =   0;
  torch::Tensor t3 = torch::zeros({Z});
  t3[0] =   1;
  t3[1] =  -1;

  // Set embeddings folder 2
  torch::Tensor t4 = torch::zeros({Z});
  t4[0] =  -2;
  t4[1] =  -2;
  torch::Tensor t5 = torch::zeros({Z});
  t5[0] =  .5;
  t5[1] =  .5;
  torch::Tensor t6 = torch::zeros({Z});
  t6[0] =   1;
  t6[1] =   1;

  d.setEmbedding(0, t1);
  d.setEmbedding(1, t2);
  d.setEmbedding(2, t3);
  d.setEmbedding(3, t4);
  d.setEmbedding(4, t5);
  d.setEmbedding(5, t6);

  REQUIRE ( d.findClosest(0, 0) == std::make_pair(1u, 1u) ); // => closest to [-1, 1] (folder1) is [.5, .5] (folder2)
  REQUIRE ( d.findClosest(0, 1) == std::make_pair(1u, 1u) ); // => closest to [ 0, 0] (folder1) is [.5, .5] (folder2)
  REQUIRE ( d.findClosest(0, 2) == std::make_pair(1u, 1u) ); // => closest to [ 1,-1] (folder1) is [.5, .5] (folder2)

  REQUIRE ( d.findClosest(1, 0) == std::make_pair(0u, 1u) ); // => closest to [-2,-2] (folder2) is [ 0,  0] (folder1)
  REQUIRE ( d.findClosest(1, 1) == std::make_pair(0u, 1u) ); // => closest to [.5,.5] (folder2) is [ 0,  0] (folder1)
  REQUIRE ( d.findClosest(1, 2) == std::make_pair(0u, 1u) ); // => closest to [ 1,-1] (folder2) is [ 0,  0] (folder1)

  // Make sure the search doesn't change the data -- results should stay the same
  REQUIRE ( d.findClosest(0, 0) == std::make_pair(1u, 1u) ); // => closest to [-1, 1] (folder1) is [.5, .5] (folder2)
  REQUIRE ( d.findClosest(0, 1) == std::make_pair(1u, 1u) ); // => closest to [ 0, 0] (folder1) is [.5, .5] (folder2)
  REQUIRE ( d.findClosest(0, 2) == std::make_pair(1u, 1u) ); // => closest to [ 1,-1] (folder1) is [.5, .5] (folder2)

  REQUIRE ( d.findClosest(1, 0) == std::make_pair(0u, 1u) ); // => closest to [-2,-2] (folder2) is [ 0,  0] (folder1)
  REQUIRE ( d.findClosest(1, 1) == std::make_pair(0u, 1u) ); // => closest to [.5,.5] (folder2) is [ 0,  0] (folder1)
  REQUIRE ( d.findClosest(1, 2) == std::make_pair(0u, 1u) ); // => closest to [ 1,-1] (folder2) is [ 0,  0] (folder1)
}
