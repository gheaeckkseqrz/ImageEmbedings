#pragma once
#include <torch/torch.h>

torch::Tensor loadImage(std::string const &path, unsigned int size = (unsigned int)-1);
