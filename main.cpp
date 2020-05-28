#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "feature_extractor.h"
#include "dataminer.h"
#include "feature_extractor.h"
#include "GUI.h"
#include "tsne.h"

int Z = 128;

struct Options : public GUIDelegate
{
public:
  virtual void increaseFileLimit()    { fileLimit++; print(); }
  virtual void decreaseFileLimit()    { if (fileLimit > 0) fileLimit--; print(); }
  virtual void increaseFolderLimit()  { folderLimit++; print(); }
  virtual void decreaseFolderLimit()  { if (folderLimit > 0) folderLimit--; print();}
  virtual void increaseMargin()       { margin += 0.01; print(); }
  virtual void decreaseMargin()       { margin -= 0.01; print(); }
  virtual void increaseSampling()     { sampling++; print(); }
  virtual void decreaseSampling()     { sampling--; print(); }
  virtual void increaseDisplayEvery() { displayEvery++; print(); }
  virtual void decreaseDisplayEvery() { if (displayEvery > 0) displayEvery--; print(); }
  void print()
  {
      std::cout << "======================" << std::endl;
      std::cout << "Folders : " << folderLimit << std::endl;
      std::cout << "Files   : " << fileLimit << std::endl;
      std::cout << "Margin  : " << margin << std::endl;
      std::cout << "Sampling  : " << sampling << std::endl;
      std::cout << "DisplayEvery  : " << displayEvery << std::endl;
      std::cout << "======================" << std::endl;
  }

  size_t fileLimit = 100;
  size_t folderLimit = 8700;
  float margin = .9;
  float sampling = 100;
  unsigned int displayEvery = 6;
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
      std::cout << "\r" << b << " / " << dataloader.size().value() / batch_size << " -- " << loss.item<float>();
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
      o.print();
      // plot(g, dataloader, model, o.folderLimit, o.fileLimit);
      plot(g, dataloader, model, 200, 100);
      for (unsigned int i(0) ; i  < o.displayEvery ; ++i)
	{
	  dataloader.updateIdEmbedings();
	  std::cout << i << " -- " << train(dataloader, model, optimizer, o.margin) << std::endl;
	  Triplet t = dataloader.get(rand());
	  g.showTriplet(dataloader.getPath(t.anchor_folder_index[0].item<int64_t>(), t.anchor_index[0].item<int64_t>()),
			dataloader.getPath(t.anchor_folder_index[0].item<int64_t>(), t.same_index[0].item<int64_t>()),
			dataloader.getPath(t.diff_folder_index[0].item<int64_t>(), t.diff_index[0].item<int64_t>()));
	}
      torch::save(model, "model.pt");
      // o.folderLimit++;
    }

  return 0;
}
