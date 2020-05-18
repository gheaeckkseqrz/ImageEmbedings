#include <torch/torch.h>
#include <iostream>
#include <cmath>

#include "feature_extractor.h"
#include "dataminer.h"
#include "feature_extractor.h"
#include "GUI.h"
#include "tsne.h"

int Z = 128;

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

  size_t fileLimit = 100;
  size_t folderLimit = 8724;
  float margin = .9;
  float sampling = 0;
  unsigned int displayEvery = 5000;
};

void plot(GUI &gui, Dataminer &dataloader, FeatureExtractor &model, size_t folder_limit, size_t file_limit)
{
  folder_limit = std::min(folder_limit, dataloader.nbIdentities());
  unsigned int N = dataloader.size(folder_limit);
  torch::Tensor tsneInput = torch::zeros({N, Z});
  torch::Tensor tsneOutput = torch::zeros({N, 2u}, torch::kDouble);
  unsigned int j(0);
  model->eval();
  for (unsigned int folder(0) ; folder < folder_limit ; ++folder)
  {
    std::cout << "\rPlot " << folder+1 << " / " << folder_limit;
    std::cout.flush();
    for (unsigned int i(0) ; i < std::min(file_limit, dataloader.identitySize(folder)) ; ++i)
      {
	torch::Tensor image = dataloader.getImage(folder, i).unsqueeze(0);
	torch::Tensor code = model->forward(image);
	// dataloader.setEmbedding(folder, i, code[0].data());
	tsneInput[j].copy_(code[0].data());
	j++;
      }
  }
  tsneInput = tsneInput.to(torch::kDouble);
  TSNE::run(tsneInput.data_ptr<double>(), N, Z, tsneOutput.data_ptr<double>(), 2, 50, .5, -1, false, 1000, 250, 250);
  j = 0;
  for (unsigned int folder(0) ; folder < folder_limit ; ++folder)
    for (unsigned int i(0) ; i < std::min(file_limit, dataloader.identitySize(folder)) ; ++i)
      {
	gui.addPoint(tsneOutput[j][0].item<float>(), tsneOutput[j][1].item<float>(), folder, dataloader.getPath(folder, i));
	j++;
      }
  gui.update();
  std::cout << std::endl;
}

float train(Dataminer &dataloader, FeatureExtractor &model, torch::optim::Adam &optimizer, float margin)
{
  model->train();
  float total_loss = 0;
  torch::Tensor reference_target = torch::zeros(std::vector<int64_t>({Z})).cuda();
  reference_target[0] = .8;
  torch::Tensor norm_target = torch::ones(std::vector<int64_t>({1})).cuda();
  norm_target.fill_(torch::norm(reference_target).item<float>());

  torch::Tensor label_pos = torch::zeros(std::vector<int64_t>({1})).cuda();
  torch::Tensor label_neg = torch::zeros(std::vector<int64_t>({1})).cuda();

  label_pos.fill_( 1);
  label_neg.fill_(-1);

  optimizer.zero_grad();
  for (unsigned int i(0) ; i < 32 ; ++i)
    {
      Triplet training_triplet = dataloader.get(0);
      torch::Tensor anchor_code = model->forward(training_triplet.anchor.unsqueeze(0)); //.detach();
      torch::Tensor same_code = model->forward(training_triplet.same.unsqueeze(0));
      torch::Tensor diff_code = model->forward(training_triplet.diff.unsqueeze(0));

      dataloader.setEmbedding(training_triplet.anchor_folder_index, training_triplet.anchor_index, anchor_code[0].data());
      dataloader.setEmbedding(training_triplet.anchor_folder_index, training_triplet.same_index,   same_code[0].data());
      dataloader.setEmbedding(training_triplet.diff_folder_index,   training_triplet.diff_index,   diff_code[0].data());

      torch::Tensor norm_same = torch::norm(same_code);
      torch::Tensor norm_diff = torch::norm(diff_code);

      torch::Tensor loss_same = torch::cosine_embedding_loss(same_code, anchor_code, label_pos, margin);
      torch::Tensor loss_diff = torch::cosine_embedding_loss(diff_code, anchor_code, label_neg, margin);
      torch::Tensor loss_norm_same = torch::relu(norm_same - 0.8);
      torch::Tensor loss_norm_diff = torch::relu(norm_diff - 0.8);

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
  //dataloader.fillCache(200, 100);
  FeatureExtractor model(32, Z);
  torch::load(model, "model.pt");
  model->to(at::kCUDA);
  torch::optim::Adam optimizer(model->parameters(), 0.0001);

  // Get a couple of point for sampling
  std::cout << "Initial pass" << std::endl;
  for (unsigned int folder(0) ; folder < std::min(o.folderLimit, dataloader.nbIdentities()) ; ++folder)
    {
      std::cout << "\r" << folder << " / " << o.folderLimit;
      std::cout.flush();
      torch::Tensor image = dataloader.getImage(folder, 0).unsqueeze(0);
      torch::Tensor code = model->forward(image);
      dataloader.setIdEmbedding(folder, code[0].data());
    }
  std::cout << std::endl;

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
      //plot(g, dataloader, model, o.folderLimit, o.fileLimit);
      plot(g, dataloader, model, 200, 100);
      for (int i(0) ; i  < o.displayEvery ; ++i)
	{
	  std::cout << i << " -- " << train(dataloader, model, optimizer, o.margin) << std::endl;
	  if (i % 200 == 0)
	    dataloader.updateIdEmbedings();
	}
      torch::save(model, "model.pt");
      // o.folderLimit++;
    }

  return 0;
}
