#pragma once

#include <torch/torch.h>

struct ResidualBlockImpl : torch::nn::Module
{
  ResidualBlockImpl(unsigned int nc)
    {
      _c1  = register_module("c1",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));
      _c2  = register_module("c2",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));
      _c3  = register_module("c3",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));
      _c4  = register_module("c4",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));

      _n1   = register_module("n1",  torch::nn::InstanceNorm2d(nc));
      _n2   = register_module("n2",  torch::nn::InstanceNorm2d(nc));
      _n3   = register_module("n3",  torch::nn::InstanceNorm2d(nc));
      _n4   = register_module("n4",  torch::nn::InstanceNorm2d(nc));
    }

  torch::Tensor forward(torch::Tensor x)
    {
      torch::Tensor base = x;
      x = _n1(torch::relu(_c1(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n2(torch::relu(_c2(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n3(torch::relu(_c3(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n4(_c4(x));
      return x + base;
    }

  torch::nn::Conv2d _c1  = nullptr;
  torch::nn::Conv2d _c2  = nullptr;
  torch::nn::Conv2d _c3  = nullptr;
  torch::nn::Conv2d _c4  = nullptr;

  torch::nn::InstanceNorm2d _n1  = nullptr;
  torch::nn::InstanceNorm2d _n2  = nullptr;
  torch::nn::InstanceNorm2d _n3  = nullptr;
  torch::nn::InstanceNorm2d _n4  = nullptr;
};

TORCH_MODULE(ResidualBlock);
