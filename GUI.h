#pragma once

#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

struct Item
{
  float x;
  float y;
  unsigned int label;
  std::string path;
};

class GUIDelegate
{
 public:
  virtual void increaseFileLimit() = 0;
  virtual void decreaseFileLimit() = 0;
  virtual void increaseFolderLimit() = 0;
  virtual void decreaseFolderLimit() = 0;
  virtual void increaseMargin() = 0;
  virtual void decreaseMargin() = 0;
  virtual void increaseSampling() = 0;
  virtual void decreaseSampling() = 0;
  virtual void increaseDisplayEvery() = 0;
  virtual void decreaseDisplayEvery() = 0;
};

class GUI
{
 public:
  GUI(GUIDelegate * delegate = nullptr)
    :_delegate(delegate) {}

  ~GUI()
  {
    _alive = false;
    _thread.join();
  }

  void start()
  {
    _thread = std::thread(&GUI::_start, this);
  }

  void addPoint(float x, float y, unsigned int label, std::string path)
  {
    Item i;
    i.x = x * 1000 + 1000;
    i.y = y * 1000 + 1000;
    i.label = label;
    i.path = std::move(path);
    _pointsNext.push_back(i);
    while (label >= _colors.size())
      _colors.push_back(sf::Color(rand() % 255, rand() % 255, rand() % 255));
  }

  void update()
  {
    std::scoped_lock lock(_pointActiveM);
    _pointsActive.swap(_pointsNext);
    _pointsNext.clear();
  }

 private:
  void _start()
  {
    _window.create(sf::VideoMode(2000, 2000), "My window");
    _alive = true;
    while (_alive)
      {
	sf::Event event;
        while (_window.pollEvent(event))
	  {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
	      {
		_window.close();
		break;
	      }
	    if (event.type == sf::Event::KeyReleased && _delegate)
	      {
		switch(event.key.code)
		  {
		  case sf::Keyboard::Q:
		    _delegate->increaseFolderLimit();
		      break;
		  case sf::Keyboard::A:
		    _delegate->decreaseFolderLimit();
		      break;
		  case sf::Keyboard::W:
		    _delegate->increaseFileLimit();
		      break;
		  case sf::Keyboard::S:
		    _delegate->decreaseFileLimit();
		      break;
		  case sf::Keyboard::E:
		    _delegate->increaseMargin();
		      break;
		  case sf::Keyboard::D:
		    _delegate->decreaseMargin();
		      break;
		  case sf::Keyboard::R:
		    _delegate->increaseSampling();
		    break;
		  case sf::Keyboard::F:
		    _delegate->decreaseSampling();
		    break;
		  case sf::Keyboard::T:
		    _delegate->increaseDisplayEvery();
		    break;
		  case sf::Keyboard::G:
		    _delegate->decreaseDisplayEvery();
		    break;
		  default:
		    break;
		  }
	      }
	  }
	sf::Vector2i mouse = sf::Mouse::getPosition(_window);
	_window.clear(sf::Color(10, 10, 10));
	{
	  std::scoped_lock lock(_pointActiveM);
	  for (Item const &item : _pointsActive)
	    {
	      sf::CircleShape shape(5.f);
	      shape.setFillColor(_colors[item.label]);
	      shape.setPosition(item.x - 5, item.y - 5);
	      if (std::pow(item.x - mouse.x, 2) + std::pow(item.y - mouse.y, 2) < 25)
		{
		  shape.setOutlineThickness(2.f);
		  shape.setOutlineColor(sf::Color(255, 0, 0));
		  sf::Sprite sprite;
		  sf::Texture texture;
		  texture.loadFromFile(item.path);
		  sprite.setTexture(texture);
		  _window.draw(sprite);
		}
	      _window.draw(shape);
	    }
	}
	_window.display();
      }
  }

 private:
  std::vector<Item>      _pointsNext;
  std::vector<Item>      _pointsActive;
  std::mutex             _pointActiveM;
  sf::RenderWindow       _window;
  std::thread            _thread;
  bool                   _alive;
  std::vector<sf::Color> _colors;
  GUIDelegate           *_delegate;
};