#pragma once

#include <cassert>
#include <vector>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include <torch/torch.h>

#include "images.h"

struct Triplet
{
  Triplet(unsigned int batch_size = 1)
  {
    anchor_folder_index = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    diff_folder_index = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    anchor_index = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    same_index = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    diff_index = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
  }

  torch::Tensor anchor;
  torch::Tensor same;
  torch::Tensor diff;

  torch::Tensor anchor_folder_index;
  torch::Tensor diff_folder_index;
  torch::Tensor anchor_index;
  torch::Tensor same_index;
  torch::Tensor diff_index;
};

class Dataloader : public torch::data::datasets::BatchDataset<Dataloader, Triplet>
{
 public:
  Dataloader(unsigned int image_resolution = 256, std::string filter = "", torch::Device device = torch::kCPU)
    :_device(device), _nb_images(0), _image_resolution(image_resolution), _filter(filter) {}

 Dataloader(std::string const &data_path, unsigned int image_resolution = 256, std::string filter = "", torch::Device device = torch::kCPU)
   :_device(device), _nb_images(0), _image_resolution(image_resolution), _filter(filter)
    {
      unsigned int min = std::numeric_limits<unsigned int>::max();
      unsigned int max = std::numeric_limits<unsigned int>::min();
      for (const auto & entry : fs::directory_iterator(data_path))
	{
	  if (fs::is_directory(entry))
	    {
	      unsigned int s = this->addFolder(entry.path());
	      min = std::min(s, min);
	      max = std::max(s, max);
	    }
	}
      std::cout << "Added " << _data.size() << " folders" << std::endl;
      std::cout << "Number of file per folder in range [" << min << "/" << max << "]" << std::endl;
      std::cout << "Total number of files " << *(Dataloader::size()) << std::endl;
      std::random_shuffle ( _data.begin(), _data.end() );
    }

  bool isImage(std::string const &path) const
  {
    if (path[path.size() - 3] == 'j' &&
	path[path.size() - 2] == 'p' &&
	path[path.size() - 1] == 'g')
      return true;
    if (path[path.size() - 3] == 'p' &&
	path[path.size() - 2] == 'n' &&
	path[path.size() - 1] == 'g')
      return true;

    return false;
  }

  virtual unsigned int addFolder(std::string const &path)
  {
    std::vector<std::string> file_list;
    for (const auto & entry : fs::directory_iterator(path))
      if (fs::is_regular_file(entry) && isImage(entry.path()))
	if (_filter.empty() || entry.path().filename() == _filter)
	  file_list.push_back(entry.path());
    _nb_images += file_list.size();
    if (!file_list.empty())
      {
	_data.emplace_back(std::move(file_list));
	return _data.back().size();
      }
    return 0;
  }

  void fillCache(unsigned int folders, unsigned int files)
  {
    for (unsigned int i(0) ; i < folders ; ++i)
      {
	std::cout << "\rFill cache " << i << " / " << folders;
	std::cout.flush();
	_cache.push_back(std::vector<torch::Tensor>());
	for (unsigned int j(0) ; j < files ; ++j)
	  _cache.back().push_back(loadImage(_data[i][j], _image_resolution).to(_device));
      }
    std::cout << std::endl;
  }

  std::string getPath(unsigned int folder, unsigned file) const
  {
    return _data[folder][file];
  }

  torch::Tensor getImage(unsigned int folder, unsigned file) const
    {
      return getImage(folder, file, _image_resolution);
    }

  torch::Tensor getImage(unsigned int folder, unsigned file, unsigned int size) const
    {
      if (folder < _cache.size() && file < _cache[folder].size())
	return _cache[folder][file];
      return loadImage(_data[folder][file], size).to(_device);
    }

  virtual Triplet get_batch(c10::ArrayRef<size_t> request)
  {
    unsigned int batch_size = request.size();
    Triplet res(batch_size);
    res.anchor = torch::zeros({batch_size, 3, static_cast<long int>(_image_resolution), static_cast<long int>(_image_resolution)}).to(_device);
      res.same = torch::zeros({batch_size, 3, static_cast<long int>(_image_resolution), static_cast<long int>(_image_resolution)}).to(_device);
    res.diff = torch::zeros({batch_size, 3, static_cast<long int>(_image_resolution), static_cast<long int>(_image_resolution)}).to(_device);
    unsigned int i(0);
    for (unsigned int _ : request)
      {
	Triplet t = get(_);
	res.anchor[i].copy_(t.anchor);
	res.same[i].copy_(t.same);
	res.diff[i].copy_(t.diff);
	res.anchor_folder_index[i] = t.anchor_folder_index[0];
	res.diff_folder_index[i] = t.diff_folder_index[0];
	res.anchor_index[i] = t.anchor_index[0];
	res.same_index[i] = t.same_index[0];
	res.diff_index[i] = t.diff_index[0];
	i++;
      }
    return res;
  }

  virtual Triplet get(size_t index)
    {
      Triplet res;
      assert(_data.size() > 1);
      res.anchor_folder_index[0] = static_cast<int64_t>(index % std::min(_data.size(), _max_folder));
      res.diff_folder_index[0] = static_cast<int64_t>(rand() % std::min(_data.size(), _max_folder));
      while (res.diff_folder_index[0].item<int64_t>() == res.anchor_folder_index[0].item<int64_t>())
	res.diff_folder_index[0] = static_cast<int64_t>(rand() % std::min(_data.size(), _max_folder));
      res.anchor_index[0] = static_cast<int64_t>(rand() % std::min(_data[res.anchor_folder_index[0].item<int64_t>()].size(), _max_file));
      res.same_index[0] = static_cast<int64_t>(rand() % std::min(_data[res.anchor_folder_index[0].item<int64_t>()].size(), _max_file));
      res.diff_index[0] = static_cast<int64_t>(rand() % std::min(_data[res.diff_folder_index[0].item<int64_t>()].size(), _max_file));

      res.anchor = getImage(res.anchor_folder_index[0].item<int64_t>(), res.anchor_index[0].item<int64_t>(), _image_resolution);
      res.same = getImage(res.anchor_folder_index[0].item<int64_t>(), res.same_index[0].item<int64_t>(), _image_resolution);
      res.diff = getImage(res.diff_folder_index[0].item<int64_t>(), res.diff_index[0].item<int64_t>(), _image_resolution);
      return res;
    }

  // Size return the number of identities, as we want a full iteration
  // over the dataset to select each identity as an anchor once.
  c10::optional<long unsigned int> size() const
  {
    return nbIdentities();
  }

  size_t size(size_t until) const
  {
    unsigned int total(0);
    for (size_t i(0) ; i < until ; ++i)
      total += std::min(_data[i].size(), _max_file);
    return total;
  }

  size_t nbIdentities() const
  {
    return _data.size();
  }

  size_t nbImages() const
  {
    return _nb_images;
  }

  size_t identitySize(size_t id) const
  {
    return _data[id].size();
  }

  void setLimits(unsigned int files, unsigned int folders)
  {
    _max_file = files;
    _max_folder = folders;
  }

  std::pair<unsigned int , unsigned int> findFolderAndFileForIndex(unsigned int index)
  {
    std::pair<unsigned int , unsigned int> res;
    res.first = 0;
    res.second = 0;
    while (index >= _data[res.first].size())
      {
	index -= _data[res.first].size();
	res.first += 1;
      }
    res.second = index;
    return res;
  }

protected:
  torch::Device _device;
  size_t _max_folder = std::numeric_limits<int>::max();
  size_t _max_file = std::numeric_limits<int>::max();
  size_t _nb_images;
  size_t _image_resolution;
  std::string _filter;
  std::vector<std::vector<std::string>> _data;
  std::vector<std::vector<torch::Tensor>> _cache;
};
