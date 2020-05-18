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

class Dataloader : public torch::data::datasets::Dataset<Dataloader, Triplet>
{
 public:
  Dataloader( unsigned int size = 256)
    :_size(size) {}

 Dataloader(std::string const &data_path, unsigned int size = 256)
   :_size(size)
    {
      unsigned int min = std::numeric_limits<unsigned int>::max();
      unsigned int max = std::numeric_limits<unsigned int>::min();
      for (const auto & entry : fs::directory_iterator(data_path))
	{
	  if (entry.is_directory())
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
      if (entry.is_regular_file() && isImage(entry.path()))
	  file_list.push_back(entry.path());
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
	  _cache.back().push_back(loadImage(_data[i][j], _size).cuda());
      }
    std::cout << std::endl;
  }

  std::string getPath(unsigned int folder, unsigned file) const
  {
    return _data[folder][file];
  }

  torch::Tensor getImage(unsigned int folder, unsigned file) const
    {
      return getImage(folder, file, _size);
    }

  torch::Tensor getImage(unsigned int folder, unsigned file, unsigned int size) const
    {
      if (folder < _cache.size() && file < _cache[folder].size())
	return _cache[folder][file];
      return loadImage(_data[folder][file], size).cuda();
    }

  virtual Triplet get(size_t index)
    {
      (void)index;
      Triplet res;
      assert(_data.size() > 1);
      res.anchor_folder_index = rand() % std::min(_data.size(), _max_folder);
      res.diff_folder_index = rand() % std::min(_data.size(), _max_folder);
      while (res.diff_folder_index == res.anchor_folder_index)
	res.diff_folder_index = rand() % std::min(_data.size(), _max_folder);
      res.anchor_index = rand() % std::min(_data[res.anchor_folder_index].size(), _max_file);
      res.same_index = rand() % std::min(_data[res.anchor_folder_index].size(), _max_file);
      res.diff_index = rand() % std::min(_data[res.diff_folder_index].size(), _max_file);

      res.anchor = getImage(res.anchor_folder_index, res.anchor_index, _size);
      res.same = getImage(res.anchor_folder_index, res.same_index, _size);
      res.diff = getImage(res.diff_folder_index, res.diff_index, _size);
      return res;
    }

  c10::optional<long unsigned int>  size() const
  {
    unsigned int total(0);
    for (auto const &folder : _data)
      total += folder.size();
    return total;
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
  size_t _max_folder;
  size_t _max_file;
  size_t _size;
  std::vector<std::vector<std::string>> _data;
  std::vector<std::vector<torch::Tensor>> _cache;
};
