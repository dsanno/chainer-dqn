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
from chainer import cuda, optimizers, serializers
import sys
from PyQt4.QtGui import QPixmap, QApplication
from PIL import Image

latent_size = 256
gamma = 0.99
batch_size = 64
ag.PAUSE = 0

parser = argparse.ArgumentParser(description='Deep Q-learning Network for game using mouse')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--interval', default=100, type=int,
                    help='interval of capturing (ms)')
parser.add_argument('--random', '-r', default=0.2, type=float,
                    help='randomness of play')
parser.add_argument('--pool_size', default=50000, type=int,
                    help='number of frames of memory pool size')
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
parser.add_argument('--double_dqn', action='store_true',
                    help='use Double DQN algorithm')
parser.add_argument('--update_target_interval', default=2000, type=int,
                    help='interval to update target Q function of Double DQN')
parser.add_argument('--only_result', action='store_true',
                    help='use only reward to evaluate')
args = parser.parse_args()

interval = args.interval / 1000.0
only_result = args.only_result
use_double_dqn = args.double_dqn
update_target_interval = args.update_target_interval
game = PoohHomerun()
game.load_images('image')
if game.detect_position() is None:
    print "Error: cannot detect game screen position."
    exit()
left, top, w, h = game.region()
train_width = w / 4
train_height = h / 4
random.seed()

gpu_device = None
xp = np
q = Q(width=train_width, height=train_height, latent_size=latent_size, action_size=game.action_size())
target_q = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device_from_id(gpu_device).use()
    xp = cuda.cupy
    q.to_gpu()

POOL_SIZE = args.pool_size
state_pool = np.zeros((POOL_SIZE, 3, train_height, train_width), dtype=np.float32)
action_pool = np.zeros((POOL_SIZE,), dtype=np.int32)
reward_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
terminal_pool = np.zeros((POOL_SIZE,), dtype=np.float32)

# allocate memory
state_pool[...] = 0
action_pool[...] = 0
reward_pool[...] = 0
terminal_pool[...] = 0

if only_result:
    terminal_pool[-1] = 1
frame = 0
average_reward = 0

optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-06)
optimizer.setup(q)
optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))
if args.input is not None:
    serializers.load_hdf5('{}.model'.format(args.input), q)
    serializers.load_hdf5('{}.state'.format(args.input), optimizer)

random_probability = args.random
random_reduction_rate = 1 - args.random_reduction
min_random_probability = min(random_probability, args.min_random)

app = QApplication(sys.argv)
window_id = app.desktop().winId()

def train():
    max_term_size = args.max_train_term
    current_term_size = args.train_term
    term_increase_rate = 1 + args.train_term_increase
    last_clock = time.clock()
    update_target_iteration = 0
    if use_double_dqn:
        target_q = q.copy()
        target_q.reset_state()
    while True:
        term_size = int(current_term_size)
        if frame < batch_size * term_size:
            continue
        batch_index = np.random.permutation(min(frame - term_size, POOL_SIZE))[:batch_size]
        train_image = xp.asarray(state_pool[batch_index])
        y = q(train_image)
        if use_double_dqn and update_target_iteration >= update_target_interval:
            target_q = q.copy()
            target_q.reset_state()
            with chainer.using_config('enable_backprop', False):
                target_q(xp.asarray(state_pool[batch_index]))
            update_iteration = 0
        for term in range(term_size):
            next_batch_index = (batch_index + 1) % POOL_SIZE
            train_image = xp.asarray(state_pool[next_batch_index])
            score = q(train_image)
            if only_result:
                t = xp.asarray(reward_pool[batch_index])
            else:
                if use_double_dqn:
                    eval_image = xp.asarray(state_pool[next_batch_index])
                    with chainer.using_config('enable_backprop', False):
                        target_score = target_q(eval_image)
                    best_action = cuda.to_cpu(xp.argmax(score.data, axis=1))
                    best_q = cuda.to_cpu(target_score.data)[range(batch_size), best_action]
                else:
                    best_q = cuda.to_cpu(xp.max(score.data, axis=1))
                t = xp.asarray(reward_pool[batch_index] + (1 - terminal_pool[batch_index]) * gamma * best_q)
            action_index = xp.asarray(action_pool[batch_index])
            loss = F.mean_squared_error(F.select_item(y, action_index), t)
            y = score
            q.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            batch_index = next_batch_index
            print "loss", float(cuda.to_cpu(loss.data))
            clock = time.clock()
            print "train", clock - last_clock
            last_clock = clock
            if use_double_dqn:
                update_target_iteration += 1
        current_term_size = min(current_term_size * term_increase_rate, max_term_size)
        print "current_term_size ", current_term_size

if __name__ == '__main__':
    try:
        thread.start_new_thread(train, ())
        next_clock = time.clock() + interval
        save_iter = 10000
        save_count = 0
        action = None
        action_q = q.copy()
        action_q.reset_state()
        while True:
            if action is not None:
                game.play(action)

            pixmap = QPixmap.grabWindow(window_id, left, top, w, h)
            image = pixmap.toImage()
            bits = image.bits()
            bits.setsize(image.byteCount())
            screen = Image.fromarray(np.array(bits).reshape((h, w, 4))[:,:,2::-1])
            reward, terminal = game.process(screen)
            if reward is not None:
                train_image = xp.asarray(screen.resize((train_width, train_height))).astype(np.float32).transpose((2, 0, 1))
                train_image = xp.expand_dims(train_image, 0) / 127.5 - 1
                with chainer.using_config('train', False):
                    with chainer.using_config('enable_backprop', False):
                        score = action_q(train_image)
                best = int(np.argmax(score.data))
                action = game.randomize_action(best, random_probability)
                print action, float(score.data[0][action]), best, float(score.data[0][best]), reward
                index = frame % POOL_SIZE
                state_pool[index] = cuda.to_cpu(train_image)
                action_pool[index] = action
                reward_pool[index - 1] = reward
                average_reward = average_reward * 0.9999 + reward * 0.0001
                print "average reward: ", average_reward
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
                if save_iter <= 0:
                    print 'save: ', save_count
                    serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, save_count), q)
                    serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, save_count), optimizer)
                    save_iter = 10000
                    save_count += 1
            current_clock = time.clock()
            wait = next_clock - current_clock
            print 'wait: ', wait
            if wait > 0:
                next_clock += interval
                time.sleep(wait)
            elif wait > -interval / 2:
                next_clock += interval
            else:
                next_clock = current_clock + interval
    except KeyboardInterrupt:
        pass
