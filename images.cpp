#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "images.h"


/*
  Loads image and return it as a tensor of shape [CxHxW]
  with channel order BGR in range [0, 1]
*/
torch::Tensor loadImage(std::string const &path)
{
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if(! image.data )
    std::cout <<  "Could not open or find the image [" << path << "]" << std::endl ;
  cv::Mat image_float;
  image.convertTo(image_float, CV_32F);
  torch::Tensor t = torch::from_blob(image_float.data, {image.rows, image.cols, 3});
  t.transpose_(2, 0);
  return t / 255;
}