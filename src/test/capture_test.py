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
parser.add_argument('--number', '-n', default=10000, type=int,
                    help='number of screenthots')
parser.add_argument('--interval', '-i', default=100, type=int,
                    help='interval of capturing (ms)')
args = parser.parse_args()
x, y, w, h = args.left, args.top, args.width, args.height
interval = args.interval

start = time.clock()
for i in range(args.number):
    for name in ['homerun', 'hit', 'foul', 'strike', 'stage', ]:
        position = ag.locateOnScreen('image/{}.png'.format(name), region=(x, y, w, h));
        if position != None:
            left, top, width, height = position
            print '{} {} {}'.format(name, left, top)
