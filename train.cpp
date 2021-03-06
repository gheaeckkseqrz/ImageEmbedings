#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataminer.h"
#include "hyperparameters.h"


constexpr unsigned int SAVE_EVERY = 100;
constexpr float MARGIN = .9;

void pretrain(FeatureExtractor &model, Dataloader &dataloader, torch::Device device, unsigned int epochs)
{
  unsigned int nbIdentities = std::min(static_cast<unsigned int>(dataloader.nbIdentities()), 1024u);
  unsigned int batch_size = std::min(12u, nbIdentities / 2);

  dataloader.setLimits(1000, nbIdentities);
  torch::nn::Sequential m(model,
			  torch::nn::Linear(Hyperparameters::Z, 2000),
			  torch::nn::ReLU(),
			  torch::nn::Linear(2000, 2000),
			  torch::nn::ReLU(),
			  torch::nn::Linear(2000, nbIdentities));
  std::cout << "Pretrain\n" << m << std::endl;
  m->to(device);
  torch::optim::Adam optimizer(m->parameters(), Hyperparameters::PRETRAIN_LR);
  m->train();

  std::ofstream pretrain_loss_file("pretrain_loss.txt");
  for (unsigned int epoch(0) ; epoch < epochs ; ++ epoch)
  {
    auto start = std::chrono::high_resolution_clock::now();
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataloader, torch::data::DataLoaderOptions().batch_size(batch_size).workers(5));
    unsigned int b(0);
    float total_loss(0);
    unsigned int accuracy(0);
    unsigned int total(0);
    for (Triplet &training_triplet : *data_loader)
    {
      if (training_triplet.anchor_folder_index.sizes()[0] != batch_size)
	break; // Drop Last
      optimizer.zero_grad();

      torch::Tensor pred = m->forward(training_triplet.anchor);
      pred = torch::log_softmax(pred, 1);

      std::tuple<at::Tensor, at::Tensor> max = torch::max(pred, 1);
      for (unsigned int b(0) ; b < batch_size ; b++)
	if (std::get<1>(max)[b].item<int>() == training_triplet.anchor_folder_index[b].item<int>())
	  accuracy++;
      total += batch_size;

      torch::Tensor loss = torch::nn::functional::nll_loss(pred, training_triplet.anchor_folder_index.cuda());
      total_loss += loss.item<float>();
      loss.backward();
      optimizer.step();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Pretrain " << epoch << " / " << epochs << " -- Duration : " << duration.count() << " seconds -- " << total_loss << " -- Top 1 Accuracy : " << accuracy << " / " << total << std::endl;
    pretrain_loss_file << total_loss << std::endl;
    if (epoch % SAVE_EVERY == 0)
      torch::save(model, "feature_extractor.pt");
  }
}

std::pair<float, float> evaluate(Dataloader const &dataloader, FeatureExtractor &model, float margin, torch::Device device)
{
  // No need for backprop in evaluation
  torch::NoGradGuard no_grad;

  model->eval();
  float total_var_loss = 0;
  float total_margin_loss = 0;

  torch::Tensor identity_codes = torch::zeros({static_cast<unsigned int>(dataloader.nbIdentities()), Hyperparameters::Z});
  for (unsigned int i(0) ; i < dataloader.nbIdentities() ; ++i)
  {
    unsigned int identity_size = std::min(size_t(12), dataloader.identitySize(i));
    torch::Tensor codes = torch::zeros({identity_size, Hyperparameters::Z}).to(device);
    for (unsigned int j(0) ; j < identity_size ; ++j)
    {
      torch::Tensor input = dataloader.getImage(i, j).unsqueeze(0);
      torch::Tensor code = model->forward(input);
      codes[j].copy_(torch::nn::functional::normalize(code.data())[0]);
    }
    torch::Tensor variance = torch::var(codes, {0}, true, false);
    torch::Tensor mean = torch::mean(codes, {0});
    total_var_loss += torch::sum(variance).item<float>();
    identity_codes[i].copy_(torch::nn::functional::normalize(mean.unsqueeze(0))[0]);
  }
  torch::Tensor margins = torch::mm(identity_codes, identity_codes.t());
  torch::Tensor triu = torch::triu(margins);
  total_margin_loss += torch::sum(triu).item<float>();
  return {total_var_loss, total_margin_loss};
}

float train(Dataminer &dataloader, FeatureExtractor &model, torch::optim::Adam &optimizer, float margin, torch::Device device, unsigned int epoch)
{
  unsigned int batch_size = std::min(1u, static_cast<unsigned int>(dataloader.nbIdentities() / 2));
  model->train();
  float total_loss = 0;

  torch::Tensor label_pos = torch::zeros(std::vector<int64_t>({batch_size})).to(device);
  torch::Tensor label_neg = torch::zeros(std::vector<int64_t>({batch_size})).to(device);
  label_pos.fill_( 1);
  label_neg.fill_(-1);

  torch::Tensor reference = torch::zeros(std::vector<int64_t>({Hyperparameters::Z})).to(device);
  reference[0] = 1;

  auto start = std::chrono::high_resolution_clock::now();
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataloader, torch::data::DataLoaderOptions().batch_size(batch_size).workers(5));
  unsigned int b(0);
  for (Triplet &training_triplet : *data_loader)
    {
      if (training_triplet.anchor_folder_index.sizes()[0] != batch_size)
	break; // Drop Last

      optimizer.zero_grad();
      torch::Tensor anchor_code = model->forward(training_triplet.anchor).detach();
      torch::Tensor same_code = model->forward(training_triplet.same);
      torch::Tensor diff_code = model->forward(training_triplet.diff);

      for (unsigned int i(0) ; i < batch_size ; ++i)
	{
	  dataloader.setEmbedding(training_triplet.anchor_folder_index[i].item<int64_t>(), training_triplet.anchor_index[i].item<int64_t>(), anchor_code[i].data());
	  dataloader.setEmbedding(training_triplet.anchor_folder_index[i].item<int64_t>(), training_triplet.same_index[i].item<int64_t>(),   same_code[i].data());
	  dataloader.setEmbedding(training_triplet.diff_folder_index[i].item<int64_t>(),   training_triplet.diff_index[i].item<int64_t>(),   diff_code[i].data());
	  if (training_triplet.anchor_folder_index[i].item<int64_t>() == 0)
	    anchor_code[i].copy_(reference);
	}

      torch::Tensor norm_same = torch::norm(same_code, 2, 1);
      torch::Tensor norm_diff = torch::norm(diff_code, 2, 1);

      torch::Tensor loss_same = torch::nn::functional::cosine_embedding_loss(same_code, anchor_code, label_pos, torch::nn::functional::CosineEmbeddingLossFuncOptions().margin(MARGIN).reduction(torch::kSum));
      torch::Tensor loss_diff = torch::nn::functional::cosine_embedding_loss(diff_code, anchor_code, label_neg, torch::nn::functional::CosineEmbeddingLossFuncOptions().margin(MARGIN).reduction(torch::kSum));
      torch::Tensor loss = loss_diff + loss_same;
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
  if (ac != 3)
    {
      std::cout << "Usage: " << av[0] << " TRAIN_DATA_ROOT TEST_DATA_ROOT" << std::endl;
      return -1;
    }

  torch::Device device(torch::kCUDA);
  Dataminer train_dataloader(Hyperparameters::Z, av[1], 256, "", device);
  Dataloader test_dataloader(av[2], 256, "", device);
  train_dataloader.setSampling(0);
  train_dataloader.fillCache(7, 12);
  test_dataloader.fillCache(7, 12);
  FeatureExtractor model(Hyperparameters::NC, Hyperparameters::Z);
  std::cout << model << std::endl;
  model->to(device);
  pretrain(model, train_dataloader, device, Hyperparameters::PRETRAIN_FOR);
  torch::optim::Adam optimizer(model->parameters(), Hyperparameters::TRAIN_LR);

  std::ofstream train_loss_file("train_loss.txt");
  std::ofstream variance_loss_file("variance_loss.txt");
  std::ofstream margin_loss_file("margin_loss.txt");
  unsigned int epoch(0);
  while (epoch < 10)
    {
      for (unsigned int i(0) ; i  < SAVE_EVERY ; ++i)
	{
	  //train_dataloader.updateIdEmbedings();
	  float train_loss = train(train_dataloader, model, optimizer, MARGIN, device, epoch);
	  // std::pair<float, float> eval_loss = evaluate(test_dataloader, model, MARGIN, device);
	  std::cout << i << " -- " << train_loss << std::endl;
	  train_loss_file << train_loss << std::endl;
	  // variance_loss_file << eval_loss.first << std::endl;
	  // margin_loss_file << eval_loss.second << std::endl;
	  epoch++;
	}
      torch::save(model, "feature_extractor.pt");
    }

  return 0;
}
