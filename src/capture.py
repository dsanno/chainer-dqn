import argparse
import time
import os
import pyautogui as ag

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--left', required=True, type=int,
                    help='left position of region')
parser.add_argument('--top', required=True, type=int,
                    help='top position of region')
parser.add_argument('--width', required=True, type=int,
                    help='width of region')
parser.add_argument('--height', required=True, type=int,
                    help='height of region')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output directory')
parser.add_argument('--number', '-n', default=10000, type=int,
                    help='number of screenthots')
parser.add_argument('--interval', '-i', default=100, type=int,
                    help='interval of capturing (ms)')
args = parser.parse_args()
x, y, w, h = args.left, args.top, args.width, args.height
interval = args.interval
out_dir = args.output
if not os.path.exists(out_dir):
    try:
        os.mkdir(out_dir)
    except:
        print 'cannot make directory {}'.format(out_dir)
        exit()
elif not os.path.isdir(out_dir):
    print 'file path {} exists but is not directory'.format(out_dir)
    exit()

start = time.clock()
for i in range(args.number):
    ag.screenshot(region=(x, y, w, h)).save(os.path.join(out_dir, 'cap_{0:07d}.png'.format(i)))
    time.sleep((i + 1) * interval / 1000.0 - (time.clock() - start))
