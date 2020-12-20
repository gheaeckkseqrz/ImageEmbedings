#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataloader.h"
#include "z.h"

#include "GUI.h"
#include "tsne.h"

#define SAMPLE_PER_CLASS 10

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

  // Display Results
  GUI ui;
  ui.start();

  // No need for backprop in the visualisation
  torch::NoGradGuard no_grad;

  torch::Device device(torch::kCUDA);
  Dataloader dataloader(av[2], 256, "", device);
  dataloader.fillCache(6, 12);
  FeatureExtractor model(32, Z);

  while (true)
  {
    auto ftime = fs::last_write_time(av[1]);
    torch::load(model, av[1]);
    model->to(device);
    model->eval();

    // Run all images through the encoder
    unsigned int i(0);
    torch::Tensor codes = torch::zeros({dataloader.nbIdentities() * SAMPLE_PER_CLASS, Z});
    for (unsigned int identity(0) ; identity < dataloader.nbIdentities() ; ++identity)
    {
      for (unsigned int image(0) ; image < SAMPLE_PER_CLASS ; ++image)
      {
	torch::Tensor input = dataloader.getImage(identity, image);
	torch::Tensor code = model->forward(input.unsqueeze(0));
	codes[i].copy_(torch::nn::functional::normalize(code)[0]);
	i++;
	std::string path = dataloader.getPath(identity, image);
      }
    }

    // Perform TSNE dimentionality reduction
    torch::Tensor projection = (Z > 2) ? tsne(codes) : codes.clone();

    unsigned int j(0);
    for (unsigned int identity(0) ; identity < dataloader.nbIdentities() ; ++identity)
    {
      for (unsigned int image(0) ; image < SAMPLE_PER_CLASS ; ++image)
      {
	float x = projection[j][0].item<float>();
	float y = projection[j][1].item<float>();
	ui.addPoint(x, y, identity, dataloader.getPath(identity, image));
	j++;
      }
    }
    ui.update();

    while (ftime == fs::last_write_time(av[1]))
      std::this_thread::sleep_for(std::chrono::seconds(1));
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Refresh!" << std::endl;
  }
  return 0;
}
