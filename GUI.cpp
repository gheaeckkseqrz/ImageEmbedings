#include "GUI.h"

int main(int ac, char **av)
{
  GUI g;
  g.start();

  g.addPoint(100, 100, 1, "/home/wilmot_p/DATA/VGG_FACES/vggface2_train/train/n000003/0163_02.jpg");
  g.addPoint(200, 100, 2, "/home/wilmot_p/DATA/VGG_FACES/vggface2_train/train/n000003/0164_02.jpg");
  g.addPoint(300, 100, 3, "/home/wilmot_p/DATA/VGG_FACES/vggface2_train/train/n000003/0165_01.jpg");
  g.addPoint(400, 100, 4, "/home/wilmot_p/DATA/VGG_FACES/vggface2_train/train/n000003/0167_02.jpg");
  g.addPoint(500, 100, 5, "/home/wilmot_p/DATA/VGG_FACES/vggface2_train/train/n000003/0168_02.jpg");
  g.update();

  getchar();
  return 0;
}
