#include <torch/torch.h>
#include <iostream>

#include "feature_extractor.h"
#include "dataloader.h"
#include "feature_extractor.h"
#include "plot.h"

void train(Dataloader &dataloader, std::shared_ptr<FeatureExtractor> model, torch::optim::Adam &optimizer)
{
  ScatterPlot p;
  for (unsigned int i(0) ; i < 100 ; ++i)
    {
      Triplet training_triplet = dataloader.getTriplet();
      torch::Tensor anchor_code = model->forward(training_triplet.anchor.unsqueeze(0)).detach();
      torch::Tensor same_code = model->forward(training_triplet.same.unsqueeze(0));
      torch::Tensor diff_code = model->forward(training_triplet.diff.unsqueeze(0));

      std::cout << anchor_code.sizes() << std::endl;
      p.addPoint(anchor_code[0][0].item<float>(), anchor_code[0][1].item<float>(), training_triplet.anchor_folder_index);
      p.addPoint(same_code[0][0].item<float>(), anchor_code[0][1].item<float>(), training_triplet.anchor_folder_index);
      p.addPoint(diff_code[0][0].item<float>(), anchor_code[0][1].item<float>(), training_triplet.diff_folder_index);

      optimizer.zero_grad();
      torch::Tensor loss_same = torch::mse_loss(same_code, anchor_code);
      torch::Tensor loss_diff = torch::mse_loss(diff_code, anchor_code);
      torch::Tensor loss = loss_same - loss_diff;
      loss.backward();
      optimizer.step();
      p.display();
      cv::waitKey(100);
    }
}

int main(int ac, char **av)
{
  if (ac != 2)
    {
      std::cout << "Usage: " << av[0] << " DATA_ROOT" << std::endl;
      return -1;
    }

  Dataloader dataloader(av[1], 256);
  auto model = std::make_shared<FeatureExtractor>(1, 2);
  torch::optim::Adam optimizer(model->parameters(), 0.01);

  train(dataloader, model, optimizer);

  cv::waitKey(0);
  return 0;
}
