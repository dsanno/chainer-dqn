import os
import random
import time
import numpy as np
from PIL import Image
import pyautogui as ag

class Game(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def load_images(self, image_dir):
        raise NotImplementedError

    def process():
        raise NotImplementedError

    def actions():
        raise NotImplementedError

    def play():
        raise NotImplementedError

    def findImage(self, screen, image, x, y, w, h):
        position = ag.locate(image, screen, region=(x, y, w, h));
        if position != None:
            return (position[0] + position[2] / 2, position[1] + position[3] / 2)
        return None

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

    def __init__(self, x, y):
        super(PoohHomerun, self).__init__(x, y)
        self.state = self.STATE_TITLE
        self.pausing_play = False
        self.images = {}
        self.width = 600
        self.height = 450

    def load_images(self, image_dir):
        for name in ['start', 'stage', 'select_title', 'select', 'end', 'homerun', 'hit', 'foul', 'strike']:
            self.images[name] = Image.open(os.path.join(image_dir, '{}.png'.format(name)))

    def process(self, screen):
        if self.state == self.STATE_TITLE:
            return self._process_title(screen)
        elif self.state == self.STATE_SELECT:
            return self._process_select(screen)
        elif self.state == self.STATE_RESULT:
            return self._process_result(screen)
        else:
            return self._process_play(screen)

    def actions(self):
        return self.ACTIONS.copy()

    def play(self, action):
        x, y, up, down = action
        self.move_to((x + 1) * self.width / 2, 300)
        if up > 0:
            self.mouseup()
        else:
            self.mousedown()

    def _process_title(self, screen):
        self.move_to(0, 0)
        position = self.findImage(screen, self.images['start'], 270, 240, 60, 40)
        if position != None:
            x, y = position
            self.move_to(x, y)
            time.sleep(0.1)
            self.click()
            time.sleep(0.1)
        position = self.findImage(screen, self.images['select_title'], 10, 16, 60, 40)
        if position != None:
            self.state = self.STATE_SELECT
        return (None, True)

    def _process_select(self, screen):
        self.move_to(0, 0)
        for i in reversed(range(8)):
            position = self.findImage(screen, self.images['stage'], 70 + i % 4 * 130, 180 + i / 4 * 170, 80, 20)
            if position != None and (i == 0 or random.randint(0, 1) == 0):
                x, y = position
                self.move_to(x, y + 10)
                time.sleep(0.1)
                self.click()
                time.sleep(0.1)
                break
        position = self.findImage(screen, self.images['select_title'], 10, 16, 60, 40)
        if position == None:
            self.state = self.STATE_PLAY
            return (None, False)
        return (None, False)

    def _process_play(self, screen):
        position = self.findImage(screen, self.images['end'], 250, 180, 100, 80)
        if position != None:
            self.mouseup()
            self.pausing_play = False
            self.state = self.STATE_RESULT
            return 0, True
        position = self.findImage(screen, self.images['homerun'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return 100, True
        position = self.findImage(screen, self.images['hit'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return -80, True
        position = self.findImage(screen, self.images['foul'], 250, 180, 100, 80)
        if position != None:
            if self.pausing_play:
                return None, False
            self.pausing_play = True
            return -90, True
        position = self.findImage(screen, self.images['strike'], 250, 180, 100, 80)
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
        position = self.findImage(screen, self.images['select'], 460, 406, 60, 40)
        if position != None:
            x, y = position
            self.move_to(x, y)
            time.sleep(0.1)
            self.click()
            time.sleep(0.1)
        position = self.findImage(screen, self.images['select_title'], 10, 16, 60, 40)
        if position != None:
            self.state = self.STATE_SELECT
        return (None, True)
