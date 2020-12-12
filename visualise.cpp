#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataloader.h"
#include "z.h"

#include "GUI.h"
#include "tsne.h"

torch::Tensor tsne(torch::Tensor tsneInput)
{
  unsigned int N = tsneInput.sizes()[0];
  torch::Tensor tsneOutput = torch::zeros({N, 2u}, torch::kDouble);

  tsneInput = tsneInput.to(torch::kDouble);
  TSNE::run(tsneInput.data_ptr<double>(), N, Z, tsneOutput.data_ptr<double>(), 2, 50, .5, -1, false, 1000, 250, 250);

  return tsneOutput;
}

int main(int ac, char **av)
{
  if (ac != 3)
    {
      std::cout << "Usage: " << av[0] << " CHECKPOINT DATA_ROOT" << std::endl;
      return -1;
    }

  // No need for backprop in the visualisation
  torch::NoGradGuard no_grad;

  torch::Device device(torch::kCUDA);
  Dataloader dataloader(av[2], 256, "", device);
  FeatureExtractor model(32, Z);
  torch::load(model, av[1]);
  model->to(device);
  model->eval();

  // Run all images through the encoder
  unsigned int i(0);
  torch::Tensor codes = torch::zeros({dataloader.nbIdentities(), Z});
  for (unsigned int identity(0) ; identity < dataloader.nbIdentities() ; ++identity)
    {
      for (unsigned int image(0) ; image < dataloader.identitySize(identity) ; ++image)
	{
	  torch::Tensor input = dataloader.getImage(identity, image);
	  torch::Tensor code = model->forward(input.unsqueeze(0));
	  codes[i].copy_(code[0]);
	  i++;
	  std::string path = dataloader.getPath(identity, image);
	  std::cout << path << std::endl;
	  break;
	}
    }

  // Perform TSNE dimentionality reduction
  torch::Tensor projection = tsne(codes);

  // Display Results
  GUI ui;
  ui.start();

  unsigned int j(0);
  for (unsigned int identity(0) ; identity < dataloader.nbIdentities() ; ++identity)
    {
      for (unsigned int image(0) ; image < dataloader.identitySize(identity) ; ++image)
	{
	  float x = projection[j][0].item<float>();
	  float y = projection[j][1].item<float>();
	  ui.addPoint(x, y, identity, dataloader.getPath(identity, image));
	  j++;
	  break;
	}
    }
  ui.update();
  getchar();
  return 0;
}
