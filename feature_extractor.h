#pragma once

#include <torch/torch.h>
#include "residual_block.h"

struct FeatureExtractorImpl : torch::nn::Module
{
  FeatureExtractorImpl(unsigned int nc, unsigned int nz)
    {
      _c1   = register_module("rgb2features", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, nc, 1).stride(1).bias(true)));
      _res1 = register_module("residual1", ResidualBlock(nc*1));
      _pool1 = register_module("pool1", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*1, nc*2, 3).padding(1).stride(2).bias(true)));
      _res2 = register_module("residual2", ResidualBlock(nc*2));
      _pool2 = register_module("pool2", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*2, nc*3, 3).padding(1).stride(2).bias(true)));
      _res3 = register_module("residual3", ResidualBlock(nc*3));
      _pool3 = register_module("pool3", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*3, nc*4, 3).padding(1).stride(2).bias(true)));
      _res4 = register_module("residual4", ResidualBlock(nc*4));
      _pool4 = register_module("pool4", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*4, nc*5, 3).padding(1).stride(2).bias(true)));
      _res5 = register_module("residual5", ResidualBlock(nc*5));

      _n1   = register_module("n1",  torch::nn::InstanceNorm2d(nc * 1));
      _n2   = register_module("n2",  torch::nn::InstanceNorm2d(nc * 1));
      _n3   = register_module("n3",  torch::nn::InstanceNorm2d(nc * 2));
      _n4   = register_module("n4",  torch::nn::InstanceNorm2d(nc * 2));
      _n5   = register_module("n5",  torch::nn::InstanceNorm2d(nc * 3));
      _n6   = register_module("n6",  torch::nn::InstanceNorm2d(nc * 3));
      _n7   = register_module("n7",  torch::nn::InstanceNorm2d(nc * 4));
      _n8   = register_module("n8",  torch::nn::InstanceNorm2d(nc * 4));
      _n9   = register_module("n9",  torch::nn::InstanceNorm2d(nc * 5));
      _n10  = register_module("n10", torch::nn::InstanceNorm2d(nc * 5));


      _f = register_module("f", torch::nn::Flatten());

      _fc1 = register_module("fc1", torch::nn::Linear(nc * 5 * 16 * 16, 256));
      _fc2 = register_module("fc2", torch::nn::Linear(256, 128));
      _fc3 = register_module("fc3", torch::nn::Linear(128, 64));
      _fc4 = register_module("fc4", torch::nn::Linear(64, nz));
    }

  torch::Tensor forward(torch::Tensor x)
    {
      x = _n1(torch::relu(_c1(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n2(torch::relu(_res1(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n3(torch::relu(_pool1(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n4(torch::relu(_res2(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n5(torch::relu(_pool2(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n6(torch::relu(_res3(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n7(torch::relu(_pool3(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n8(torch::relu(_res4(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n9(torch::relu(_pool4(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n10(torch::relu(_res5(x)));

      x = _f(x);

      x = torch::relu(_fc1(x));
      x = torch::relu(_fc2(x));
      x = torch::relu(_fc3(x));
      x = _fc4(x);
      x = torch::tanh(x);

      return x;
    }

  torch::nn::Conv2d _c1  = nullptr;
  torch::nn::Conv2d _pool1  = nullptr;
  torch::nn::Conv2d _pool2  = nullptr;
  torch::nn::Conv2d _pool3  = nullptr;
  torch::nn::Conv2d _pool4  = nullptr;
  ResidualBlock _res1 = nullptr;
  ResidualBlock _res2 = nullptr;
  ResidualBlock _res3 = nullptr;
  ResidualBlock _res4 = nullptr;
  ResidualBlock _res5 = nullptr;
  torch::nn::InstanceNorm2d _n1  = nullptr;
  torch::nn::InstanceNorm2d _n2  = nullptr;
  torch::nn::InstanceNorm2d _n3  = nullptr;
  torch::nn::InstanceNorm2d _n4  = nullptr;
  torch::nn::InstanceNorm2d _n5  = nullptr;
  torch::nn::InstanceNorm2d _n6  = nullptr;
  torch::nn::InstanceNorm2d _n7  = nullptr;
  torch::nn::InstanceNorm2d _n8  = nullptr;
  torch::nn::InstanceNorm2d _n9  = nullptr;
  torch::nn::InstanceNorm2d _n10 = nullptr;


  torch::nn::Flatten _f = nullptr;

  torch::nn::Linear _fc1 = nullptr;
  torch::nn::Linear _fc2 = nullptr;
  torch::nn::Linear _fc3 = nullptr;
  torch::nn::Linear _fc4 = nullptr;
};

TORCH_MODULE(FeatureExtractor);
