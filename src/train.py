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

POOL_SIZE = 10000 * 10
latent_size = 256
gamma = 0.98
batch_size = 32
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
parser.add_argument('--random_reduction', default=0.000002, type=float,
                    help='reduction rate of randomness')
parser.add_argument('--min_random', default=0.1, type=float,
                    help='minimum randomness of play')
parser.add_argument('--train_term', default=4, type=int,
                    help='training term size')
parser.add_argument('--train_term_increase', default=0.00002, type=float,
                    help='increase rate of training term size')
parser.add_argument('--max_train_term', default=32, type=int,
                    help='maximum training term size')
parser.add_argument('--only_result', default=0, type=int, choices=[0, 1],
                    help='use only reward to evaluate')
args = parser.parse_args()

interval = args.interval / 1000.0
left = args.left
top = args.top
w = args.width
h = args.height
only_result = args.only_result == 1
game = PoohHomerun(left, top)
game.load_images('image')
train_width = w / 4
train_height = h / 4
random.seed()

gpu_device = None
xp = np
actions = game.actions()
q = Q(width=train_width, height=train_height, latent_size=latent_size, action_size=len(actions))
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy
    q.to_gpu()

state_pool = np.zeros((POOL_SIZE, 3, train_height, train_width), dtype=np.float32)
action_pool = np.zeros((POOL_SIZE,), dtype=np.int32)
reward_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
terminal_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
if only_result:
    terminal_pool[-1] = 1
frame = 0

optimizer = optimizers.AdaDelta(rho=0.95, eps=0.01)
optimizer.setup(q)
optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))
if args.input is not None:
    serializers.load_hdf5('{}.model'.format(args.input), q)
    serializers.load_hdf5('{}.state'.format(args.input), optimizer)

random_probability = args.random / 20
random_reduction_rate = 1 - args.random_reduction
min_random_probability = min(random_probability, args.min_random / 20)
random_count = 0
random_min_count = 10
random_max_count = 30
random_button = True
random_position = True


def train():
    max_term_size = args.max_train_term
    current_term_size = args.train_term
    term_increase_rate = 1 + args.train_term_increase
    last_clock = time.clock()
    while True:
        term_size = int(current_term_size)
        if frame < batch_size * term_size:
            continue
        batch_index = np.random.permutation(min(frame - term_size, POOL_SIZE))[:batch_size]
        train_image = Variable(xp.asarray(state_pool[batch_index]))
        y = q(train_image)
        for term in range(term_size):
            next_batch_index = (batch_index + 1) % POOL_SIZE
            train_image = Variable(xp.asarray(state_pool[next_batch_index]))
            score = q(train_image)
            if only_result:
                t = Variable(xp.asarray(reward_pool[batch_index]))
            else:
                best_q = np.max(cuda.to_cpu(score.data), axis=1)
                t = Variable(xp.asarray(reward_pool[batch_index] + (1 - terminal_pool[batch_index]) * gamma * best_q))
            action_index = chainer.Variable(xp.asarray(action_pool[batch_index]))
            loss = F.mean_squared_error(F.select_item(y, action_index), t)
            y = score
            optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            batch_index = next_batch_index
            print "loss", float(cuda.to_cpu(loss.data))
            clock = time.clock()
            print "train", clock - last_clock
            last_clock = clock
        current_term_size = min(current_term_size * term_increase_rate, max_term_size)
        print "current_term_size ", current_term_size

if __name__ == '__main__':
    try:
        thread.start_new_thread(train, ())
        next_clock = time.clock() + interval
        save_iter = 5000
        save_count = 0
        action_size = len(actions)
        action = None
        action_q = q.copy()
        action_q.reset_state()
        while True:
            if action is not None:
                game.play(action)
            screen = ag.screenshot(region=(left, top, w, h))
            reward, terminal = game.process(screen)
            if reward is not None:
                train_image = xp.asarray(screen.resize((train_width, train_height))).astype(np.float32).transpose((2, 0, 1))
                train_image = Variable(train_image.reshape((1,) + train_image.shape) / 127.5 - 1, volatile=True)
                score = action_q(train_image, train=False)

                best = int(np.argmax(score.data))
                if random.random() < random_probability:
                    random_count += random.randint(random_min_count, random_max_count)
                    random_type = random.randint(0, 2)
                    random_button = True
                    random_position = True
                    if random_type == 0:
                        random_button = False
                    elif random_type == 1:
                        random_position = False
                    prev_pos = random.randint(0, action_size // 2 - 1)
                if random_count > 0:
                    pos = best // 2
                    button = best % 2
                    if random_position:
                        pos = prev_pos + random.randint(-5, 5)
                        if pos < 0:
                            pos = 0
                        elif pos >= action_size // 2:
                            pos = action_size // 2 - 1
                        prev_pos = pos
                    if random_button:
                        button = random.randint(0, 1)
                    actual = pos * 2 + button
                    random_count -= 1
                else:
                    actual = best
                print float(actions[actual][0]), float(score.data[0][actual]), float(actions[best][0]), int(actions[best][3]), float(score.data[0][best]), reward
                action = actions[actual]
                index = frame % POOL_SIZE
                state_pool[index] = cuda.to_cpu(train_image.data)
                action_pool[index] = actual
                reward_pool[index - 1] = reward
                if terminal:
                    terminal_pool[index - 1] = 1
                    if only_result:
                        i = index - 2
                        r = reward
                        while terminal_pool[i] == 0:
                            r = reward_pool[i] + gamma * r
                            reward_pool[i] = r
                            i -= 1
                    action_q = q.copy()
                    action_q.reset_state()
                else:
                    terminal_pool[index - 1] = 0
                frame += 1
                save_iter -= 1
                random_probability *= random_reduction_rate
                if random_probability < min_random_probability:
                    random_probability = min_random_probability
            else:
                action = None
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
