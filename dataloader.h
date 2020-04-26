#pragma once

#include <cassert>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

#include <torch/torch.h>

#include "images.h"

struct Triplet
{
  torch::Tensor anchor;
  torch::Tensor same;
  torch::Tensor diff;
};

class Dataloader
{
 public:
  Dataloader(unsigned int size = 256)
    :_size(size) {}

 Dataloader(std::string const &data_path, unsigned int size = 256)
   :_size(size)
    {
      unsigned int min = std::numeric_limits<unsigned int>::max();
      unsigned int max = std::numeric_limits<unsigned int>::min();
      for (const auto & entry : fs::directory_iterator(data_path))
	{
	  unsigned int s = addFolder(entry.path());
	  min = std::min(s, min);
	  max = std::max(s, max);
	}
      std::cout << "Added " << _data.size() << " folders" << std::endl;
      std::cout << "Number of file per folder in range [" << min << "/" << max << "]" << std::endl;
    }

  unsigned int addFolder(std::string const &path)
  {
    std::vector<std::string> file_list;
    for (const auto & entry : fs::directory_iterator(path))
      file_list.push_back(entry.path());
    _data.emplace_back(std::move(file_list));
    return _data.back().size();
  }

  torch::Tensor get(unsigned int folder, unsigned file) const
    {
      return get(folder, file, _size);
    }

  torch::Tensor get(unsigned int folder, unsigned file, unsigned int size) const
    {
      return loadImage(_data[folder][file], size);
    }

  Triplet getTriplet() const
    {
      assert(_data.size() > 1);
      unsigned int anchor_folder_index = rand() % _data.size();
      unsigned int diff_folder_index = rand() % _data.size();
      while (diff_folder_index == anchor_folder_index)
	diff_folder_index = rand() % _data.size();
      unsigned int anchor_index = rand() % _data[anchor_folder_index].size();
      unsigned int same_index = rand() % _data[anchor_folder_index].size();
      unsigned int diff_index = rand() % _data[diff_folder_index].size();
      Triplet res;
      res.anchor = get(anchor_folder_index, anchor_index, _size);
      res.same = get(anchor_folder_index, same_index, _size);
      res.diff = get(diff_folder_index, diff_index, _size);
      return res;
    }

 private:
  unsigned int _size;
  std::vector<std::vector<std::string>> _data;
};
