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
	_embedings.push_back(torch::zeros({static_cast<long int>(folder.size()), _Z}));
      _idEmbedings = torch::zeros({static_cast<long int>(_data.size()), Z});
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

  void setIdEmbedding(unsigned int folder, torch::Tensor const &embedding)
  {
    _embedings[folder].copy_(embedding);
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

  unsigned int findClosestIdentity(unsigned int folder, unsigned int file) const
  {
    torch::Tensor target = _embedings[folder][file].unsqueeze(0).cuda();
    torch::Tensor t = _idEmbedings.slice(0, 0, _max_folder, 1).clone().cuda();
    t[folder].copy_(target[0] * -1);
    torch::Tensor cosine_similarity = at::cosine_similarity(target, t);
    std::tuple<at::Tensor, at::Tensor> max = torch::max(cosine_similarity, 0);
    return std::get<1>(max).item<int>();
  }

  unsigned int findFurthestPoint(unsigned int folder, unsigned int file) const
  {
    torch::Tensor target = _embedings[folder][file].unsqueeze(0).cuda();
    torch::Tensor t = _embedings[folder].slice(0, 0, _max_file, 1).clone().cuda();
    torch::Tensor cosine_similarity = at::cosine_similarity(target, t);
    std::tuple<at::Tensor, at::Tensor> min = torch::min(cosine_similarity, 0);
    return std::get<1>(min).item<int>();
  }

  Triplet get(size_t index) override
  {
    (void)index;
    if (rand() % 100 >= _sampling)
      return Dataloader::get(index);

    Triplet res;
    assert(_data.size() > 1);
    res.anchor_folder_index[0] = static_cast<int64_t>(index % std::min(_data.size(), _max_folder));
    res.anchor_index[0] = static_cast<int64_t>(rand() % std::min(_data[res.anchor_folder_index[0].item<int64_t>()].size(), _max_file));
    res.same_index[0] = static_cast<int64_t>(findFurthestPoint(res.anchor_folder_index[0].item<int64_t>(), res.anchor_index[0].item<int64_t>()));

    res.diff_folder_index[0] = static_cast<int64_t>(findClosestIdentity(res.anchor_folder_index[0].item<int64_t>(), res.anchor_index[0].item<int64_t>()));
    res.diff_index[0] = static_cast<int64_t>(rand() % std::min(_data[res.diff_folder_index[0].item<int64_t>()].size(), _max_file));

    res.anchor = getImage(res.anchor_folder_index[0].item<int64_t>(), res.anchor_index[0].item<int64_t>(), _size);
    res.same = getImage(res.anchor_folder_index[0].item<int64_t>(), res.same_index[0].item<int64_t>(), _size);
    res.diff = getImage(res.diff_folder_index[0].item<int64_t>(), res.diff_index[0].item<int64_t>(), _size);
    return res;
  }

  void setSampling(float s)
  {
    _sampling = s;
  }

  void updateIdEmbedings()
  {
    std::cout << "updateIdEmbedings" << std::endl;
    unsigned int i(0);
    for (torch::Tensor const &idEmbedings : _embedings)
      {
	if (_max_folder > 0 && i > _max_folder)
	  break;
	std::cout << "\r" << i << " / " << _embedings.size();
	std::cout.flush();
	_idEmbedings[i].copy_(torch::mean(idEmbedings, 0).data());
	i++;
      }
    std::cout << std::endl;
  }

 private:
  std::vector<torch::Tensor> _embedings;
  torch::Tensor              _idEmbedings;
  unsigned int               _Z;
  float                      _sampling;
};
