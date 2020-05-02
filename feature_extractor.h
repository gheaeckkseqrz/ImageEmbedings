#pragma once

#include <torch/torch.h>

struct FeatureExtractor : torch::nn::Module
{
  FeatureExtractor(unsigned int nc, unsigned int nz)
    {
      _c1  = register_module("c1",  torch::nn::Conv2d(torch::nn::Conv2dOptions(  3,  nc*1, 3).padding(1).stride(1).bias(true)));
      _c2  = register_module("c2",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*1, nc*1, 3).padding(1).stride(1).bias(true)));
      _c3  = register_module("c3",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*1, nc*2, 3).padding(1).stride(2).bias(true)));
      _c4  = register_module("c4",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*2, nc*2, 3).padding(1).stride(1).bias(true)));
      _c5  = register_module("c5",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*2, nc*2, 3).padding(1).stride(1).bias(true)));
      _c6  = register_module("c6",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*2, nc*4, 3).padding(1).stride(2).bias(true)));
      _c7  = register_module("c7",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*4, nc*4, 3).padding(1).stride(1).bias(true)));
      _c8  = register_module("c8",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*4, nc*4, 3).padding(1).stride(1).bias(true)));
      _c9  = register_module("c9",  torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*4, nc*8, 3).padding(1).stride(2).bias(true)));
      _c10 = register_module("c10", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*8, nc*8, 3).padding(1).stride(1).bias(true)));
      _c11 = register_module("c11", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*8, nc*8, 3).padding(1).stride(1).bias(true)));
      _c12 = register_module("c12", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*8, nc*8, 3).padding(1).stride(2).bias(true)));
      _c13 = register_module("c13", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc*8, nc*8, 3).padding(1).stride(1).bias(true)));

      _n1   = register_module("n1",  torch::nn::InstanceNorm2d(nc * 1));
      _n2   = register_module("n2",  torch::nn::InstanceNorm2d(nc * 1));
      _n3   = register_module("n3",  torch::nn::InstanceNorm2d(nc * 2));
      _n4   = register_module("n4",  torch::nn::InstanceNorm2d(nc * 2));
      _n5   = register_module("n5",  torch::nn::InstanceNorm2d(nc * 2));
      _n6   = register_module("n6",  torch::nn::InstanceNorm2d(nc * 4));
      _n7   = register_module("n7",  torch::nn::InstanceNorm2d(nc * 4));
      _n8   = register_module("n8",  torch::nn::InstanceNorm2d(nc * 4));
      _n9   = register_module("n9",  torch::nn::InstanceNorm2d(nc * 8));
      _n10  = register_module("n10", torch::nn::InstanceNorm2d(nc * 8));
      _n11  = register_module("n11", torch::nn::InstanceNorm2d(nc * 8));
      _n12  = register_module("n12", torch::nn::InstanceNorm2d(nc * 8));
      _n13  = register_module("n13", torch::nn::InstanceNorm2d(nc * 8));

      _f = register_module("f", torch::nn::Flatten());

      _fc1 = register_module("fc1", torch::nn::Linear(nc * 8 * 16 * 16, 256));
      _fc2 = register_module("fc2", torch::nn::Linear(256, 128));
      _fc3 = register_module("fc3", torch::nn::Linear(128, 64));
      _fc4 = register_module("fc4", torch::nn::Linear(64, nz));
    }

  ~FeatureExtractor() = default;

  torch::Tensor forward(torch::Tensor x)
    {
      x = _n1(torch::relu(_c1(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n2(torch::relu(_c2(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n3(torch::relu(_c3(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n4(torch::relu(_c4(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n5(torch::relu(_c5(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n6(torch::relu(_c6(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n7(torch::relu(_c7(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n8(torch::relu(_c8(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n9(torch::relu(_c9(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n10(torch::relu(_c10(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n11(torch::relu(_c11(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n12(torch::relu(_c12(x)));
      x = torch::dropout(x, 0.2, is_training());
      x = _n13(torch::relu(_c13(x)));

      x = _f(x);

      x = torch::relu(_fc1(x));
      x = torch::relu(_fc2(x));
      x = torch::relu(_fc3(x));
      x = _fc4(x);
      x = torch::tanh(x);

      return x;
    }

  torch::nn::Conv2d _c1  = nullptr;
  torch::nn::Conv2d _c2  = nullptr;
  torch::nn::Conv2d _c3  = nullptr;
  torch::nn::Conv2d _c4  = nullptr;
  torch::nn::Conv2d _c5  = nullptr;
  torch::nn::Conv2d _c6  = nullptr;
  torch::nn::Conv2d _c7  = nullptr;
  torch::nn::Conv2d _c8  = nullptr;
  torch::nn::Conv2d _c9  = nullptr;
  torch::nn::Conv2d _c10 = nullptr;
  torch::nn::Conv2d _c11 = nullptr;
  torch::nn::Conv2d _c12 = nullptr;
  torch::nn::Conv2d _c13 = nullptr;
  torch::nn::Conv2d _c14 = nullptr;

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
  torch::nn::InstanceNorm2d _n11 = nullptr;
  torch::nn::InstanceNorm2d _n12 = nullptr;
  torch::nn::InstanceNorm2d _n13 = nullptr;

  torch::nn::Flatten _f = nullptr;

  torch::nn::Linear _fc1 = nullptr;
  torch::nn::Linear _fc2 = nullptr;
  torch::nn::Linear _fc3 = nullptr;
  torch::nn::Linear _fc4 = nullptr;
};
