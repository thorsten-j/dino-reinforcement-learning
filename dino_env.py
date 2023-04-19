import random
import time

class Obstacle:
  """
  An obstacle is a rectangular box with width and height
  and it has a y position (0 for cactus, > 0 for bird).
  Obstacles move by decreasing their distance by 1. Birds
  do not additionally move in x direction, they behave
  like a flying cactus.
  """
  def __init__(self, distance, y, w, h):
    self.distance = distance
    self.w = w
    self.h = h
    self.y = y

NUMBER_OF_PLAYER_ACTIONS = 3
NUMBER_OF_ENV_ACTIONS = 9 # 8 obstacles + pass

JUMP_PHASES = []

STAND = 0
CROUCH = 1
JUMP = 2

ENV_PASS_ACTION = 3

JUMP_PHASES = [
  0, 1, 2, 3, 2, 1
]

OBSTACLES = [
  [0, 1,2], # big cactus
  [0, 1,1], # small cactus
  [0, 2,2], # two big cactus 
  [0, 2,1], # two small cactus
  [0, 3,2], # three big cactus
  [0, 3,1], # three small cactus
  [1, 1,1], # low bird
  [2, 1,1], # high bird
]

class State:
  """
  Legal actions for player:
  0: stand
  1: crouch
  2: jump

  Crouching starts immediately and lasts
  as long as crouch action is chosen. Jump
  starts immediately but lasts for some
  frames and during an active jump, the
  only possible action is jump. All other
  actions are considered legal but do not
  affect state.

  To be a MDP, acting player alters between
  player and environment. Environment
  actions are:
  3: nothing happens
  >= 4: obstacle appears (see OBSTACLES)
  Obstacles appear 40 time steps ahead.  
  """
  def __init__(self):
    self.y = 0 # dino on ground
    self.h = 2 # not crouching
    self.jump_phase = -1 # not jumping
    self.obstacles = [self._create_obstacle(1)]
    self.speed = 100 # 100 ms per step
    self.at_move = 1 # player acts first
    self.score = 0
    self.terminal = False

  def clone(self):
    s = State()
    s.y = self.y
    s.h = self.h
    s.jump_phase = self.jump_phase
    s.speed = self.speed
    s.at_move = self.at_move
    s.score = self.score
    s.terminal = self.terminal
    s.obstacles = []
    for o in self.obstacles:
      s.obstacles.append(Obstacle(o.distance, o.y, o.w, o.h))
    return s

  def _player_action(self, action):
    if self.at_move == 0:
      raise Exception("player action but environment is at move")
    if self.jump_phase == -1: # else: ignore action
      self.h = 2
      if action == CROUCH:
        self.h = 1
      elif action == JUMP:
        self.jump_phase = 0 # start to jump
        self.y = JUMP_PHASES[self.jump_phase]
    self.at_move = 0 # env next

  def _env_action(self, action):
    if self.at_move == 1:
      raise Exception("environment action but player is at move")
    if action != ENV_PASS_ACTION:
      od = OBSTACLES[action - 4]
      self.obstacles.append(Obstacle(40, od[0], od[1], od[2]))
    # generic stuff: jump progress, time progress
    if self.jump_phase != -1:
      self.jump_progress()
    self.time_progress()
    self.at_move = 1 # player next

  def jump_progress(self):
    self.jump_phase += 1
    if self.jump_phase == len(JUMP_PHASES):
      self.y = 0
      self.jump_phase = -1
    else:
      self.y = JUMP_PHASES[self.jump_phase]

  def time_progress(self):
    obstacles = self.obstacles
    self.obstacles = []
    for obstacle in obstacles:
      obstacle.distance -= 1
      # detect collision
      if obstacle.distance <= 0 and obstacle.distance + obstacle.w - 1 >= 0:
        y1 = obstacle.y
        y2 = obstacle.y + obstacle.h - 1
        dino_y1 = self.y
        dino_y2 = self.y + self.h - 1
        if (y1 >= dino_y1 and y1 <= dino_y2) or (y2 >= dino_y1 and y2 <= dino_y2):
          # crash
          self.terminal = True
      if obstacle.distance + obstacle.w + 1 > 0: 
        self.obstacles.append(obstacle)
    if not self.terminal:
      self.score += 1

  def step(self, action):
    s_ = self.clone()
    s_.apply_action(action)
    # step function automatically takes care of
    # environment action
    if not s_.terminal:
      s_.apply_action(s_.choose_random_env_action())
    r = -1.0 if s_.terminal else 0.1
    # experiment: punishment for jumps
    if action == JUMP:
      r = 0
    return s_, r, s_.terminal

  def apply_action(self, action):
    if self.terminal:
      raise Exception("apply_action on terminal state not possible")
    if action < 0:
      raise Exception("action out of bounds")
    elif action < NUMBER_OF_PLAYER_ACTIONS:
      self._player_action(action)
    elif action < NUMBER_OF_PLAYER_ACTIONS + NUMBER_OF_ENV_ACTIONS:
      self._env_action(action)
    else:
      raise Exception("action out of bounds")

  def _create_obstacle(self, max_offset):
    idx = random.randint(0, max_offset)
    obstacle_data = OBSTACLES[idx]
    return Obstacle(40, obstacle_data[0], obstacle_data[1], obstacle_data[2])

  def choose_random_env_action(self):
    if self.at_move != 0:
      raise Exception("env action but env is not at move")
    # random choice strategy:
    # up to 200 points: 0,1, distance 15-25
    # up to 300 points: 0,1,2,3, distance 10-15
    # up to 400 points: 0-6, distance 5-15
    # more than 500 points: all actions, distance 10-20
    if self.score < 200:
      max = 1
      min_dist = 15
      max_dist = 25
    elif self.score < 300:
      max = 3
      min_dist = 10
      max_dist = 15
    elif self.score < 400:
      max = 5
      min_dist = 5
      max_dist = 15
    else:
      max = 7
      min_dist = 10
      max_dist = 20
    
    # New obstacles appear in distance 40 from dino.
    # They appear if and only if distance to last
    # obstacle is within min_dist and max_dist.
    current_dist = 40 - (self.obstacles[-1].distance + self.obstacles[-1].w - 1)
    if current_dist >= min_dist and current_dist < max_dist:
      # It may or may not appear.
      # Probability is 1/(max_dist - current_dist), so for
      # example 1/10 if there are 10 distance points left
      # for obstacle to appear.
      dist_rand = random.randint(0, max_dist - current_dist)
      if dist_rand == 0:
        return random.randint(0, max) + ENV_PASS_ACTION
      else:
        return ENV_PASS_ACTION
    elif current_dist >= max_dist:
      # it will appear
      return random.randint(0, max) + ENV_PASS_ACTION
    else:
      return ENV_PASS_ACTION # no new obstacle

  def str(self):
    s = list(
"""                                             #
                                             #
                                             #
                                             #
.............................................#
""")
    # dino at x=3
    for i in range(self.h):
      s[(4-(self.y+i))*47 + 3] = 'D'
    # obstacles
    for obstacle in self.obstacles:
      for i in range(obstacle.h):
        for j in range(obstacle.w):
          if obstacle.y == 0: # cactus
            s[(4-(obstacle.y+i))*47 + 3 + obstacle.distance + j] = '|'
          else: # bird
            s[(4-(obstacle.y+i))*47 + 3 + obstacle.distance + j] = '<'
    out = ""
    for c in s:
      out += c
    return out

def play_dino():
  game = State()
  while not game.terminal:
    if game.at_move == 1:
      v = input()
      if v:
        action = int(v)
      else:
        action = 0
      game.apply_action(action)
    else:
      action = game.choose_random_env_action()
      game.apply_action(action)
    if game.at_move == 0:
      print(game.str())
      print(game.score)
      # sleep
      time.sleep(game.speed / 1000)

if __name__ == "__main__":
  play_dino()

