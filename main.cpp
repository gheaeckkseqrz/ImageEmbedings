#include <torch/torch.h>
#include <iostream>
#include <cmath>

#include "feature_extractor.h"
#include "dataminer.h"
#include "feature_extractor.h"
#include "plot.h"

std::vector<cv::Scalar_<int>> ScatterPlot::_colors;
int Z = 2;

void plot(Dataminer &dataloader, std::shared_ptr<FeatureExtractor> model, unsigned int folder_limit, unsigned int file_limit)
{
  model->eval();
  ScatterPlot p;
  for (unsigned int folder(0) ; folder < folder_limit ; ++folder)
  {
    for (unsigned int i(0) ; i < file_limit ; ++i)
      {
	torch::Tensor image = dataloader.get(folder, i).unsqueeze(0);
	torch::Tensor code = model->forward(image);
	dataloader.setEmbedding(folder, i, code[0].data());
	p.addPoint(code[0][0].item<float>(), code[0][1].item<float>(), folder);
      }
  }
  p.display();
  cv::waitKey(100);
}

float train(Dataminer &dataloader, std::shared_ptr<FeatureExtractor> model, torch::optim::Adam &optimizer)
{
  model->train();
  float total_loss = 0;
  torch::Tensor reference_target = torch::zeros(std::vector<int64_t>({Z})).cuda();
  reference_target[0] = 7;
  torch::Tensor norm_target = torch::ones(std::vector<int64_t>({1})).cuda();
  norm_target.fill_(torch::norm(reference_target).item<float>());

  torch::Tensor label_pos = torch::zeros(std::vector<int64_t>({1})).cuda();
  torch::Tensor label_neg = torch::zeros(std::vector<int64_t>({1})).cuda();

  label_pos.fill_( 1);
  label_neg.fill_(-1);

  optimizer.zero_grad();
  for (unsigned int i(0) ; i < 32 ; ++i)
    {
      Triplet training_triplet = dataloader.getTriplet();
      torch::Tensor anchor_code = model->forward(training_triplet.anchor.unsqueeze(0)).detach() * 10;
      torch::Tensor same_code = model->forward(training_triplet.same.unsqueeze(0)) * 10;
      torch::Tensor diff_code = model->forward(training_triplet.diff.unsqueeze(0)) * 10;

      dataloader.setEmbedding(training_triplet.anchor_folder_index, training_triplet.anchor_index, anchor_code[0].data());
      dataloader.setEmbedding(training_triplet.anchor_folder_index, training_triplet.same_index, same_code[0].data());
      dataloader.setEmbedding(training_triplet.diff_folder_index, training_triplet.diff_index, diff_code[0].data());

      torch::Tensor norm_same = torch::norm(same_code);
      torch::Tensor norm_diff = torch::norm(diff_code);

      torch::Tensor loss_same = torch::cosine_embedding_loss(same_code, anchor_code, label_pos, .5);
      torch::Tensor loss_diff = torch::cosine_embedding_loss(diff_code, anchor_code, label_neg, .5);
      torch::Tensor loss_norm_same = torch::mse_loss(norm_same, norm_target) / 500;
      torch::Tensor loss_norm_diff = torch::mse_loss(norm_diff, norm_target) / 500;

      torch::Tensor loss = loss_same + loss_diff + loss_norm_same + loss_norm_diff;
      total_loss += loss.item<float>();
      loss.backward();
    }
  optimizer.step();
  return total_loss;
}

int main(int ac, char **av)
{
  if (ac != 2)
    {
      std::cout << "Usage: " << av[0] << " DATA_ROOT" << std::endl;
      return -1;
    }

  unsigned int folder_limit = 10;
  unsigned int file_limit = 3;
  Dataminer dataloader(Z, av[1], 256);
  dataloader.fillCache(folder_limit, file_limit);
  auto model = std::make_shared<FeatureExtractor>(32, 2);
  model->to(at::kCUDA);
  torch::optim::Adam optimizer(model->parameters(), 0.0001);

  dataloader.setLimits(10, folder_limit);
  while (true)
    {
      for (int i(0) ; i  < 1 ; ++i)
      	std::cout << folder_limit << " -- " << i << " -- " << train(dataloader, model, optimizer) << std::endl;
      plot(dataloader, model, folder_limit, file_limit);
    }

  cv::waitKey(0);
  return 0;
}
