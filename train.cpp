#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataminer.h"

constexpr unsigned int Z = 512;
constexpr unsigned int SAVE_EVERY = 1;
constexpr float MARGIN = .9;

float train(Dataminer &dataloader, FeatureExtractor &model, torch::optim::Adam &optimizer, float margin)
{
  unsigned int batch_size = 6;
  model->train();
  float total_loss = 0;
  torch::Tensor reference_target = torch::zeros(std::vector<int64_t>({Z})).cuda();
  reference_target[0] = .8;
  torch::Tensor norm_target = torch::ones(std::vector<int64_t>({1})).cuda();
  norm_target.fill_(torch::norm(reference_target).item<float>());

  torch::Tensor label_pos = torch::zeros(std::vector<int64_t>({batch_size})).cuda();
  torch::Tensor label_neg = torch::zeros(std::vector<int64_t>({batch_size})).cuda();

  label_pos.fill_( 1);
  label_neg.fill_(-1);

  auto start = std::chrono::high_resolution_clock::now();
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataloader, torch::data::DataLoaderOptions().batch_size(batch_size).workers(5));
  unsigned int b(0);
  for (Triplet &training_triplet : *data_loader)
    {
      if (training_triplet.anchor_folder_index.sizes()[0] != batch_size)
	break; // Drop Last

      optimizer.zero_grad();
      torch::Tensor anchor_code = model->forward(training_triplet.anchor);
      torch::Tensor same_code = model->forward(training_triplet.same);
      torch::Tensor diff_code = model->forward(training_triplet.diff);

      for (unsigned int i(0) ; i < batch_size ; ++i)
	{
	  dataloader.setEmbedding(training_triplet.anchor_folder_index[i].item<int64_t>(), training_triplet.anchor_index[i].item<int64_t>(), anchor_code[i].data());
	  dataloader.setEmbedding(training_triplet.anchor_folder_index[i].item<int64_t>(), training_triplet.same_index[i].item<int64_t>(),   same_code[i].data());
	  dataloader.setEmbedding(training_triplet.diff_folder_index[i].item<int64_t>(),   training_triplet.diff_index[i].item<int64_t>(),   diff_code[i].data());
	}

      torch::Tensor norm_same = torch::norm(same_code, 2, 1);
      torch::Tensor norm_diff = torch::norm(diff_code, 2, 1);

      torch::Tensor loss_same = torch::cosine_embedding_loss(same_code, anchor_code, label_pos, margin);
      torch::Tensor loss_diff = torch::cosine_embedding_loss(diff_code, anchor_code, label_neg, margin);
      torch::Tensor loss_norm_same = torch::mean(torch::relu(norm_same - 0.8));
      torch::Tensor loss_norm_diff = torch::mean(torch::relu(norm_diff - 0.8));

      torch::Tensor loss = loss_same + loss_diff + loss_norm_same + loss_norm_diff;
      std::cout << "\r" << b << " / " << dataloader.nbIdentities() / batch_size << " -- " << loss.item<float>();
      std::cout.flush();
      b++;
      total_loss += loss.item<float>();
      loss.backward();
      optimizer.step();
    }
  std::cout << std::endl;
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::cout << "Duration : " << duration.count() << " seconds" << std::endl;
  return total_loss;
}

int main(int ac, char **av)
{
  if (ac != 2)
    {
      std::cout << "Usage: " << av[0] << " DATA_ROOT" << std::endl;
      return -1;
    }

  Dataminer dataloader(Z, av[1], 256);
  //dataloader.fillCache(200, 100);
  FeatureExtractor model(32, Z);
  //torch::load(model, "model.pt");
  model->to(at::kCUDA);
  torch::optim::Adam optimizer(model->parameters(), 0.0001);

  // Get a couple of point for sampling
  std::cout << "Initial pass" << std::endl;
  for (unsigned int folder(0) ; folder < dataloader.nbIdentities() ; ++folder)
    {
      std::cout << "\r" << folder << " / " << dataloader.nbIdentities();
      std::cout.flush();
      torch::Tensor image = dataloader.getImage(folder, 0).unsqueeze(0);
      torch::Tensor code = model->forward(image);
      dataloader.setIdEmbedding(folder, code[0].data());
    }
  std::cout << std::endl;

  while (true)
    {
      for (unsigned int i(0) ; i  < SAVE_EVERY ; ++i)
	{
	  dataloader.updateIdEmbedings();
	  float loss = train(dataloader, model, optimizer, MARGIN);
	  std::cout << i << " -- " << loss << std::endl;
	}
      torch::save(model, "feature_extractor.pt");
    }

  return 0;
}
