import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data
import time


def test(args):
  p.connect(p.GUI)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  fileName = os.path.join("mjcf", args.mjcf)
  print("fileName")
  print(fileName)
  bot=p.loadMJCF(fileName)
  for i in range(4):
    print(p.getJointInfo(bot[1],i))
    print(p.getJointState(bot[1], i))
  while (1):
    p.stepSimulation()
    p.getCameraImage(320, 240)
    time.sleep(0.01)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--mjcf', help='MJCF filename', default="inverted_pendulum.xml")
  args = parser.parse_args()
  test(args)
