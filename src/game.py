import os
import random
import time
import numpy as np
from PIL import Image
import pyautogui as ag

class Game(object):
    def __init__(self, width, height):
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def region(self):
        return (self.x, self.y, self.width, self.height)

    def load_images(self, image_dir):
        raise NotImplementedError

    def detect_position(self):
        raise NotImplementedError

    def process(self, screen):
        raise NotImplementedError

    def action_size(self):
        raise NotImplementedError

    def play(self, action):
        raise NotImplementedError

    def randomize_action(self, action, random_probability):
        if random.random() < random_probability:
            return random.randint(0, self.action_size() - 1)
        return action

    def find_image(self, screen, image, x=0, y=0, w=None, h=None, center=False):
        if w is None:
            w = screen.width
        if h is None:
            h = screen.height
        position = ag.locate(image, screen, region=(x, y, w, h))
        if position != None:
            if center:
                return (position[0] + position[2] / 2, position[1] + position[3] / 2)
            else:
                return (position[0], position[1])
        return None

    def find_image_center(self, screen, image, x=0, y=0, w=None, h=None):
        return self.find_image(screen, image, x, y, w, h, center=True)

    def move_to(self, x, y):
        ag.moveTo(x + self.x, y + self.y)

    def click(self):
        ag.click()

    def mousedown(self):
        ag.mouseDown()

    def mouseup(self):
        ag.mouseUp()

class PoohHomerun(Game):
    STATE_TITLE  = 0
    STATE_SELECT = 1
    STATE_PLAY   = 2
    STATE_RESULT = 3
    WIDTH        = 600
    HEIGHT       = 450
    ACTIONS      = np.array([[np.float32(i + 260) / WIDTH * 2 - 1, 0, j, 1 - j] for i in range(0, 100, 3) for j in range(2)], dtype=np.float32)
    RANDOM_ACTION_MIN_FRAME = 10
    RANDOM_ACTION_MAX_FRAME = 30

    def __init__(self, x, y):
        super(PoohHomerun, self).__init__(self.WIDTH, self.HEIGHT)
        self.state = self.STATE_TITLE
        self.pausing_play = False
        self.images = {}
        self.random = {
            'count': 0,
            'button': True,
            'position': True,
            'prev_pos': 0,
        }

    def load_images(self, image_dir):
        for name in ['start', 'stage', 'select_title', 'select', 'end', 'homerun', 'hit', 'foul', 'strike']:
            self.images[name] = Image.open(os.path.join(image_dir, '{}.png'.format(name)))

    def detect_position(self):
        screen = ag.screenshot()
        for name, offset_x, offset_y in [('start', 288, 252), ('select_title', 28, 24)]:
            position = self.find_image(screen, self.images[name])
            if position != None:
                x, y = position
                x -= offset_x
                y -= offset_y
                self.set_position(x, y)
                return (x, y)
        return None

    def process(self, screen):
        if self.state == self.STATE_TITLE:
            return self._process_title(screen)
        elif self.state == self.STATE_SELECT:
            return self._process_select(screen)
        elif self.state == self.STATE_RESULT:
            return self._process_result(screen)
        else:
            return self._process_play(screen)

    def action_size(self):
        return len(self.ACTIONS)

    def play(self, action):
        x, y, up, down = self.ACTIONS[action]
        self.move_to((x + 1) * self.WIDTH / 2, 300)
        if up > 0:
            self.mouseup()
        else:
            self.mousedown()

    def randomize_action(self, action, random_probability):
        random_count = self.random['count']
        random_button = self.random['button']
        random_position = self.random['position']
        prev_pos = self.random['prev_pos']
        pos_size = self.action_size() // 2
        if random.random() < random_probability * 2 / (self.RANDOM_ACTION_MIN_FRAME + self.RANDOM_ACTION_MAX_FRAME):
            random_count += random.randint(self.RANDOM_ACTION_MIN_FRAME, self.RANDOM_ACTION_MAX_FRAME)
            random_type = random.randint(0, 2)
            random_button = True
            random_position = True
            if random_type == 0:
                random_button = False
            elif random_type == 1:
                random_position = False
            prev_pos = random.randint(0, pos_size - 1)
        if random_count <= 0:
            return action
        pos = action // 2
        button = action % 2
        if random_position:
            pos = prev_pos + random.randint(-5, 5)
            if pos < 0:
                pos = 0
            elif pos >= pos_size:
                pos = pos_size - 1
            prev_pos = pos
        if random_button:
            button = random.randint(0, 1)
        self.random['count'] = random_count - 1
        self.random['button'] = random_button
        self.random['position'] = random_position
        self.random['prev_pos'] = prev_pos
        return pos * 2 + button

    def _process_title(self, screen):
        self.move_to(0, 0)
        time.sleep(0.1)
        position = self.find_image_center(screen, self.images['start'], 270, 240, 60, 40)
        if position != None:
            x, y = position
            self.move_to(x, y)
            time.sleep(0.1)
            self.click()
            time.sleep(3)
        position = self.find_image_center(screen, self.images['select_title'], 10, 16, 60, 40)
        if position != None:
            self.state = self.STATE_SELECT
        return (None, False)

    def _process_select(self, screen):
        self.move_to(0, 0)
        time.sleep(0.1)
        for i in reversed(range(8)):
            position = self.find_image_center(screen, self.images['stage'], 70 + i % 4 * 130, 180 + i / 4 * 170, 80, 20)
            if position != None and (i == 0 or random.randint(0, 1) == 0):
                x, y = position
                self.move_to(x, y + 10)
                time.sleep(0.1)
                self.click()
                time.sleep(2)
                break
        position = self.find_image_center(screen, self.images['select_title'], 10, 16, 60, 40)
        if position == None:
            self.state = self.STATE_PLAY
            return (None, False)
        return (None, False)

    def _process_play(self, screen):
        position = self.find_image_center(screen, self.images['end'], 250, 180, 100, 80)
        if position != None:
            self.mouseup()
            self.pausing_play = False
            self.state = self.STATE_RESULT
            return 0, True
        position = self.find_image_center(screen, self.images['homerun'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return 100, True
        position = self.find_image_center(screen, self.images['hit'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return -80, True
        position = self.find_image_center(screen, self.images['foul'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return -90, True
        position = self.find_image_center(screen, self.images['strike'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return -100, True
        if self.pausing_play:
            self.pausing_play = False
        return 0, False

    def _process_result(self, screen):
        self.move_to(0, 0)
        time.sleep(0.1)
        position = self.find_image_center(screen, self.images['select'], 460, 406, 60, 40)
        if position != None:
            x, y = position
            self.move_to(x, y)
            time.sleep(0.1)
            self.click()
            time.sleep(3)
        position = self.find_image_center(screen, self.images['select_title'], 10, 16, 60, 40)
        if position != None:
            self.state = self.STATE_SELECT
        return (None, False)
