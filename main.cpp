#include <torch/torch.h>
#include <iostream>
#include <cmath>

#include "feature_extractor.h"
#include "dataminer.h"
#include "feature_extractor.h"
#include "GUI.h"

int Z = 3;

struct Options : public GUIDelegate
{
public:
  virtual void increaseFileLimit() { fileLimit++; }
  virtual void decreaseFileLimit() { fileLimit--; }
  virtual void increaseFolderLimit() { folderLimit++; }
  virtual void decreaseFolderLimit() { folderLimit--; }
  virtual void increaseMargin() { margin += 0.01; }
  virtual void decreaseMargin() { margin -= 0.01; }
  virtual void increaseSampling() { sampling++; }
  virtual void decreaseSampling() { sampling--; }
  virtual void increaseDisplayEvery() { displayEvery++; }
  virtual void decreaseDisplayEvery() { displayEvery--; }

  unsigned int fileLimit = 100;
  unsigned int folderLimit = 3;
  float margin = .5;
  float sampling = 50;
  unsigned int displayEvery = 1;
};

void plot(GUI &gui, Dataminer &dataloader, std::shared_ptr<FeatureExtractor> model, unsigned int folder_limit, unsigned int file_limit)
{
  model->eval();
  for (unsigned int folder(0) ; folder < folder_limit ; ++folder)
  {
    for (unsigned int i(0) ; i < file_limit ; ++i)
      {
	torch::Tensor image = dataloader.get(folder, i).unsqueeze(0);
	torch::Tensor code = model->forward(image);
	dataloader.setEmbedding(folder, i, code[0].data());
	gui.addPoint(code[0][0].item<float>(), code[0][1].item<float>(), folder, dataloader.getPath(folder, i));
      }
  }
  gui.update();
}

float train(Dataminer &dataloader, std::shared_ptr<FeatureExtractor> model, torch::optim::Adam &optimizer, float margin)
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

      torch::Tensor loss_same = torch::cosine_embedding_loss(same_code, anchor_code, label_pos, margin);
      torch::Tensor loss_diff = torch::cosine_embedding_loss(diff_code, anchor_code, label_neg, margin);
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

  Options o;
  GUI g(&o);
  g.start();
  Dataminer dataloader(Z, av[1], 256);
  dataloader.fillCache(20, 100);
  auto model = std::make_shared<FeatureExtractor>(32, Z);
  model->to(at::kCUDA);
  torch::optim::Adam optimizer(model->parameters(), 0.0001);

  while (true)
    {
      dataloader.setLimits(o.fileLimit, o.folderLimit);
      dataloader.setSampling(o.sampling);
      std::cout << "======================" << std::endl;
      std::cout << "Folders : " << o.folderLimit << std::endl;
      std::cout << "Files   : " << o.fileLimit << std::endl;
      std::cout << "Margin  : " << o.margin << std::endl;
      std::cout << "Sampling  : " << o.sampling << std::endl;
      std::cout << "DisplayEvery  : " << o.displayEvery << std::endl;
      std::cout << "======================" << std::endl;
      for (int i(0) ; i  < o.displayEvery ; ++i)
      	std::cout << i << " -- " << train(dataloader, model, optimizer, o.margin) << std::endl;
      plot(g, dataloader, model, o.folderLimit, o.fileLimit);
    }

  return 0;
}
