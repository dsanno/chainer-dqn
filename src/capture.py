import argparse
import time
import os
import pyautogui as ag
from game import PoohHomerun

parser = argparse.ArgumentParser(description='Capturing images')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output directory')
parser.add_argument('--number', '-n', default=10000, type=int,
                    help='number of screenthots')
parser.add_argument('--interval', '-i', default=100, type=int,
                    help='interval of capturing (ms)')
args = parser.parse_args()

game = PoohHomerun()
game.load_images('image')
if game.detect_position() is None:
    print "Error: cannot detect game screen position."
    exit()
x, y, w, h = game.region()
train_width = w / 4
train_height = h / 4
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
images = []
for i in range(args.number):
    images.append(ag.screenshot(region=(x, y, w, h)).resize((train_width, train_height)))
    wait = (i + 1) * interval / 1000.0 - (time.clock() - start)
    print wait
    if wait > 0:
        time.sleep(wait)
for i, image in enumerate(images):
    image.save(os.path.join(out_dir, 'cap_{0:07d}.png'.format(i)))
