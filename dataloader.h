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

  unsigned int anchor_folder_index;
  unsigned int diff_folder_index;
  unsigned int anchor_index;
  unsigned int same_index;
  unsigned int diff_index;
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
      Triplet res;
      assert(_data.size() > 1);
      res.anchor_folder_index = rand() % std::min(_data.size(), _max_folder);
      res.diff_folder_index = rand() % std::min(_data.size(), _max_folder);
      while (res.diff_folder_index == res.anchor_folder_index)
	res.diff_folder_index = rand() % std::min(_data.size(), _max_folder);
      res.anchor_index = rand() % _data[res.anchor_folder_index].size();
      res.same_index = rand() % std::min(_data[res.anchor_folder_index].size(), _max_file);
      res.diff_index = rand() % std::min(_data[res.diff_folder_index].size(), _max_file);

      res.anchor = get(res.anchor_folder_index, res.anchor_index, _size);
      res.same = get(res.anchor_folder_index, res.same_index, _size);
      res.diff = get(res.diff_folder_index, res.diff_index, _size);
      return res;
    }

  void setLimits(unsigned int files, unsigned int folders)
  {
    _max_file = files;
    _max_folder = folders;
  }

 private:
  size_t _max_folder;
  size_t _max_file;
  size_t _size;
  std::vector<std::vector<std::string>> _data;
};
