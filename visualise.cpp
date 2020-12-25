#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataloader.h"
#include "z.h"

#include "GUI.h"
#include "tsne.h"

#define SAMPLE_PER_CLASS 100u

torch::Tensor tsne(torch::Tensor tsneInput)
{
  unsigned int N = tsneInput.sizes()[0];
  torch::Tensor tsneOutput = torch::zeros({N, 2u}, torch::kDouble);

  tsneInput = tsneInput.to(torch::kDouble);
  TSNE::run(tsneInput.data_ptr<double>(), N, Z, tsneOutput.data_ptr<double>(), 2, 50, .5, -1, false, 1000, 250, 250);

  return tsneOutput;
}

struct Options : public GUIDelegate
{
public:
  virtual void increaseFileLimit() { file_limit++; };
  virtual void decreaseFileLimit() { file_limit--; };
  virtual void increaseFolderLimit() { folder_limit++; };
  virtual void decreaseFolderLimit() { folder_limit--; };
  virtual void increaseMargin() {};
  virtual void decreaseMargin() {};
  virtual void increaseSampling() {};
  virtual void decreaseSampling() {};
  virtual void increaseDisplayEvery() {};
  virtual void decreaseDisplayEvery() {};

  unsigned int file_limit = 100;
  unsigned int folder_limit = 500;
};

int main(int ac, char **av)
{
  if (ac != 3)
  {
    std::cout << "Usage: " << av[0] << " CHECKPOINT DATA_ROOT" << std::endl;
    return -1;
  }

  // Display Results
  Options options;
  GUI ui(&options);
  ui.start();

  // No need for backprop in the visualisation
  torch::NoGradGuard no_grad;

  torch::Device device(torch::kCUDA);
  Dataloader dataloader(av[2], 256, "", device);
  // dataloader.fillCache(7, 12);
  FeatureExtractor model(NC, Z);

  auto ftime = fs::last_write_time(av[1]);
  torch::Tensor codes = torch::zeros({dataloader.nbIdentities() * SAMPLE_PER_CLASS, Z});
  torch::Tensor projection = torch::zeros({dataloader.nbIdentities() * SAMPLE_PER_CLASS, 2});
  while (true)
  {
    try
    {
      torch::load(model, av[1]);
      ftime = fs::last_write_time(av[1]);


      model->to(device);
      model->eval();

      // Run all images through the encoder
      unsigned int i(0);
      for (unsigned int identity(0) ; identity < dataloader.nbIdentities() ; ++identity)
      {
	for (unsigned int image(0) ; image < SAMPLE_PER_CLASS ; ++image)
	{
	  torch::Tensor input = dataloader.getImage(identity, image);
	  torch::Tensor code = model->forward(input.unsqueeze(0));
	  // codes[i].copy_(torch::nn::functional::normalize(code)[0]);
	  codes[i].copy_(code[0]);
	  i++;
	}
      }
      // Perform TSNE dimentionality reduction
      projection = (Z > 2) ? tsne(codes) : codes.clone();
    }
    catch (c10::Error const &e)
    {
    }

    do
    {
      unsigned int identity_limit = std::min(options.folder_limit, static_cast<unsigned int>(dataloader.nbIdentities()));
      for (unsigned int identity(0) ; identity < identity_limit ; ++identity)
      {
	unsigned int image_limit = std::min(std::min(SAMPLE_PER_CLASS, options.file_limit), static_cast<unsigned int>(dataloader.identitySize(identity)));
	for (unsigned int image(0) ; image < image_limit ; ++image)
	{
	  float x = projection[identity * SAMPLE_PER_CLASS + image][0].item<float>();
	  float y = projection[identity * SAMPLE_PER_CLASS + image][1].item<float>();
	  ui.addPoint(x, y, identity, dataloader.getPath(identity, image));
	}
      }
      ui.update();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }  while (ftime == fs::last_write_time(av[1]));
    std::cout << "Refresh!" << std::endl;
  }
  return 0;
}
