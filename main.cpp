#include <torch/torch.h>
#include <iostream>

#include "feature_extractor.h"
#include "images.h"

int main()
{
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
