import argparse
import time
import thread
import os
import random
import numpy as np
import pyautogui as ag
from game import PoohHomerun
from net import Q
import chainer
from chainer import functions as F
from chainer import cuda, Variable, optimizers, serializers

import gc
import objgraph

POOL_SIZE = 2000 * 10
ACTION_SIZE = 4
gamma = 0.98
batch_size = 64
ag.PAUSE = 0

parser = argparse.ArgumentParser(description='Deep Q-learning Network for game using mouse')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--left', '-l', required=True, type=int,
                    help='left position of region')
parser.add_argument('--top', '-t', required=True, type=int,
                    help='top position of region')
parser.add_argument('--width', default=640, type=int,
                    help='width of region')
parser.add_argument('--height', default=480, type=int,
                    help='height of region')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--interval', default=100, type=int,
                    help='interval of capturing (ms)')
parser.add_argument('--random', '-r', default=0.2, type=float,
                    help='randomness of play')
args = parser.parse_args()

interval = args.interval / 1000.0
left = args.left
top = args.top
w = args.width
h = args.height
game = PoohHomerun(left, top)
game.load_images('image')
train_width = w / 4
train_height = h / 4
random.seed()
use_mc = False

gpu_device = None
xp = np
actions = game.actions()
q = Q(width=train_width, height=train_height, latent_size=100, action_size=len(actions))
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy
    q.to_gpu()

state_pool = np.zeros((POOL_SIZE, 3, train_height, train_width), dtype=np.float32)
action_pool = np.zeros((POOL_SIZE,), dtype=np.int32)
prev_pool = np.zeros((POOL_SIZE, q.latent_size), dtype=np.float32)
reward_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
best_q_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
terminal_pool = np.ones((POOL_SIZE,), dtype=np.float32)
frame = 0

optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9)
optimizer.setup(q)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
if args.input != None:
    serializers.load_hdf5('{}.model'.format(args.input), q)
    serializers.load_hdf5('{}.state'.format(args.input), optimizer)

random_probability = args.random / 20
random_count = 0
random_min_count = 10
random_max_count = 30


def train():
    last_clock = time.clock()
    while True:
        if frame < batch_size + 2:
            continue
        clock = time.clock()
        print "train", clock - last_clock
        last_clock = clock
        prev_batch_index = np.random.permutation(min(frame - 2, POOL_SIZE))[:batch_size]
        batch_index = (prev_batch_index + 1) % POOL_SIZE
        next_batch_index = (batch_index + 1) % POOL_SIZE

        prev_train_image = Variable(xp.asarray(state_pool[prev_batch_index]))
        prev_train_prev = Variable(xp.asarray(prev_pool[prev_batch_index]))
        y_prev, train_prev = q((prev_train_image, prev_train_prev))
        prev_pool[batch_index] = cuda.to_cpu(train_prev.data)

        train_image = Variable(xp.asarray(state_pool[batch_index]))
        y, train_prev = q((train_image, train_prev))
        prev_pool[next_batch_index] = cuda.to_cpu(train_prev.data)

        train_image = Variable(xp.asarray(state_pool[next_batch_index]))
        score, p = q((train_image, train_prev))
        best_q_pool[next_batch_index] = np.max(cuda.to_cpu(score.data), axis=1)

        t = Variable(xp.asarray(reward_pool[batch_index] + terminal_pool[batch_index] * gamma * best_q_pool[next_batch_index]))
        action_index = chainer.Variable(xp.asarray(action_pool[batch_index]))
        loss = F.mean_squared_error(F.select_item(y, action_index), t)
        if xp.max(t.data) > 5000 or xp.min(t.data) < -5000:
            print t.data
            print loss.data
            exit()
        optimizer.zero_grads()
        loss.backward()
        print "loss", float(cuda.to_cpu(loss.data))
        optimizer.update()

if __name__ == '__main__':
    try:
        thread.start_new_thread(train, ())
        next_clock = time.clock() + interval
        prev = Variable(xp.zeros((1, q.latent_size)).astype(np.float32), volatile=True)
        save_iter = 5000
        save_count = 0
        prev_random = 0
        action_size = len(actions)
        while True:
            screen = ag.screenshot(region=(left, top, w, h))
            reward, terminal = game.process(screen)
            if reward != None:
                train_image = xp.asarray(screen.resize((train_width, train_height))).astype(np.float32).transpose((2, 0, 1))
                train_image = Variable(train_image.reshape((1,) + train_image.shape) / 127.5 - 1, volatile=True)
                score, prev = q((train_image, prev), train=False)

                best = int(np.argmax(score.data))
                if random.random() < random_probability:
                    random_count += random.randint(random_min_count, random_max_count)
                if random_count > 0:
                    prev_random = prev_random + random.randint(-10, 10)
                    if prev_random < 0:
                        prev_random = (prev_random + action_size) % 2
                    elif prev_random >= action_size:
                        prev_random = prev_random % 2 + action_size - 2
                    actual = prev_random
                    random_count -= 1
                else:
                    actual = best
                print float(actions[actual][0]), float(score.data[0][actual]), float(actions[best][0]), int(actions[best][3]), float(score.data[0][best]), reward
                action = actions[actual]
                game.play(action)
                index = frame % POOL_SIZE
                state_pool[index] = cuda.to_cpu(train_image.data)
                prev_pool[index] = cuda.to_cpu(prev.data[0])
                action_pool[index] = actual
                reward_pool[index - 1] = reward
                if terminal:
                    terminal_pool[index - 1] = 0
                    prev_random = random.randint(0, action_size - 1)
                    prev.data.fill(0)
                else:
                    terminal_pool[index - 1] = 1
                prev_action = action
                frame += 1
                save_iter -= 1
            else:
                if terminal:
                    time.sleep(2)
                if save_iter <= 0:
                    print 'save: ', save_count
                    serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, save_count), q)
                    serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, save_count), optimizer)
                    save_iter = 5000
                    save_count += 1
            current_clock = time.clock()
            wait = next_clock - current_clock
            print 'wait: ', wait
            if wait > 0:
                next_clock += interval
                time.sleep(wait)
            else:
                next_clock = current_clock + interval
    except KeyboardInterrupt:
        pass
