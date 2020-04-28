#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ScatterPlot
{
 public:
  ScatterPlot(std::string const &name="")
    :_name(name)
    {
      _canvas = cv::Mat(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 0));
      _min = std::numeric_limits<float>::max();
      _max = std::numeric_limits<float>::min();
      _margin = 50;


      _colors.push_back({  0,   0, 255});
      _colors.push_back({  0, 255,   0});
      _colors.push_back({  0, 255, 255});
      _colors.push_back({255,   0,   0});
      _colors.push_back({255,   0, 255});
      _colors.push_back({255, 255,   0});
    }

  void display()
  {
    _canvas = 0;
    cv::Point_<float> tl = scalePoint({-1, -1});
    cv::Point_<float> tr = scalePoint({ 1, -1});
    cv::Point_<float> bl = scalePoint({-1,  1});
    cv::Point_<float> br = scalePoint({ 1,  1});

    cv::line(_canvas, tl, tr, {120, 120, 120}, 2);
    cv::line(_canvas, bl, br, {120, 120, 120}, 2);
    cv::line(_canvas, tl, bl, {120, 120, 120}, 2);
    cv::line(_canvas, tr, br, {120, 120, 120}, 2);
    cv::circle(_canvas, scalePoint({0, 0}), 10, {120, 120, 120}, -1);
    cv::circle(_canvas, scalePoint({0, 0}), scalePoint({sqrt(.5), 0}).x / 2, {120, 120, 120}, 2);

    for (unsigned int i(0) ; i < _points.size() ; ++i)
      cv::circle(_canvas, scalePoint(_points[i]), 5, _colors[_labels[i]], -1);
    cv::imshow(_name, _canvas);
  }

  void addPoint(float x, float y, unsigned int label)
  {
    _min = -1; // std::min(std::min(x, y), _min);
    _max =  1; // std::max(std::max(x, y), _max);
    _points.push_back({x, y});
    _labels.push_back(label);
    while (_colors.size() <= label)
      _colors.push_back({rand() % 255, rand() % 255, rand() % 255});
  }

private:
  cv::Point_<float> scalePoint(cv::Point_<float> p)
  {
    p.x -= _min; // Move to range [0, inf]
    p.x /= (_max - _min); // Move to range [0, 1]
    p.x *= (1000 - (2 * _margin)); // Move to range [0, 900]
    p.x += _margin; // Move to range [50, 950]
    p.y -= _min; // Move to range [0, inf]
    p.y /= (_max - _min); // Move to range [0, 1]
    p.y *= (1000 - (2 * _margin)); // Move to range [0, 900]
    p.y += _margin; // Move to range [50, 950]
    return p;
  }

private:
  cv::Mat _canvas;
  std::vector<cv::Point_<float>> _points;
  std::vector<unsigned int> _labels;
  std::vector<cv::Scalar_<int>> _colors;
  std::string _name;
  float _min;
  float _max;
  int _margin;
};
