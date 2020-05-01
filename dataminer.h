#pragma once

#include "dataloader.h"

class Dataminer : public Dataloader
{
 public:
 Dataminer(unsigned int Z, unsigned int size = 256)
   :Dataloader(size), _Z(Z)
    {
    }

 Dataminer(unsigned int Z, std::string const &data_path, unsigned int size = 256)
   :Dataloader(data_path, size), _Z(Z)
    {
      for (std::vector<std::string> const &folder : _data)
	_embedings.push_back(torch::zeros({folder.size(), _Z}));
    }

  virtual unsigned int addFolder(std::string const &path) override
  {
    unsigned int ret = Dataloader::addFolder(path);
    _embedings.push_back(torch::zeros({ret, _Z}));
    return ret;
  }

  void setEmbedding(unsigned int index, torch::Tensor const &embedding)
  {
    std::pair<unsigned int, unsigned int> folder_file = findFolderAndFileForIndex(index);
    setEmbedding(folder_file.first, folder_file.second, embedding);
  }

  void setEmbedding(unsigned int folder, unsigned int file, torch::Tensor const &embedding)
  {
    _embedings[folder][file].copy_(embedding);
  }

  torch::Tensor getEmbedding(unsigned int index)
  {
    std::pair<unsigned int, unsigned int> folder_file = findFolderAndFileForIndex(index);
    return _embedings[folder_file.first][folder_file.second];
  }

  std::pair<unsigned int, unsigned int> findClosest(unsigned int folder, unsigned int file) const
  {
    torch::Tensor target = _embedings[folder][file];
    float closest = std::numeric_limits<float>::max();
    std::pair<unsigned int, unsigned int> best;
    best.first = 0;
    best.second = 0;
    for (unsigned int i(0) ; i < std::min(_embedings.size(), _max_folder) ; ++i)
      {
	if (i != folder)
	  {
	    torch::Tensor t = _embedings[i].clone();
	    t = t.slice(0, 0, _max_file, 1);
	    t -= target;
	    t = t * t;
	    t = torch::sum(t, 1);
	    std::tuple<at::Tensor, at::Tensor> min = torch::min(t, 0);
	    if (std::get<0>(min).item<float>() < closest)
	      {
		best.first = i;
		best.second = std::get<1>(min).item<int>();
		closest = std::get<0>(min).item<float>();
	      }
	  }
      }
    return best;
  }

  Triplet getTriplet() const override
  {
    if (rand() % 3 == 0)
      return Dataloader::getTriplet();

    Triplet res;
    assert(_data.size() > 1);
    res.anchor_folder_index = rand() % std::min(_data.size(), _max_folder);
    res.anchor_index = rand() % std::min(_data[res.anchor_folder_index].size(), _max_file);
    res.same_index = rand() % std::min(_data[res.anchor_folder_index].size(), _max_file);

    std::pair<unsigned int, unsigned int> diff = findClosest(res.anchor_folder_index, res.anchor_index);
    res.diff_folder_index = diff.first;
    res.diff_index = diff.second;

    res.anchor = get(res.anchor_folder_index, res.anchor_index, _size);
    res.same = get(res.anchor_folder_index, res.same_index, _size);
    res.diff = get(res.diff_folder_index, res.diff_index, _size);
    return res;
  }

 private:
  std::vector<torch::Tensor> _embedings;
  unsigned int _Z;
};