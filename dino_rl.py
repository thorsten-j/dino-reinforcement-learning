import dino_env
import tensorflow as tf
import numpy as np
import random
import time

class ReplayBuffer:
  def __init__(self, max_capacity=1000000):
    self.buffer = []
    self.capazity = max_capacity
  
  def append(self, entry):
    if len(self.buffer) >= self.capazity:
      self.buffer = self.buffer[1:]
    self.buffer.append(entry)

class DinoTrainer:
  def __init__(self):
    self.online_model = self.init_model()
    self.target_model = self.init_model()
    self.target_model.set_weights(self.online_model.get_weights())
    self.epsilon = 1.0
    self.epsilon_decrease = 0.0025
    self.epsilon_minimum = 0.01
    self.gamma = 0.9
    self.target_update_frequency = 10
    self.batch_size = 32
    self.replay_buffer = ReplayBuffer()

  def init_model(self):
    model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(93,)),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(
      loss=tf.keras.losses.MSE,
      optimizer=tf.keras.optimizers.Adam()
    )
    return model

  def as_tensor(self, state):
    # Converts state into tensor (as numpy array).
    # Tensor contains next two obstacles in form of
    # distance, x, width and height
    tensor = np.zeros((93,), dtype=np.float32)
    o1 = state.obstacles[0]
    tensor[max(o1.distance-1, 0)] = 1
    tensor[40] = o1.y
    tensor[41] = o1.w
    tensor[42] = o1.h
    if len(state.obstacles) > 1:
      o2 = state.obstacles[1]
      tensor[43+ max(o2.distance-1, 0)] = 1
      tensor[83] = o2.y
      tensor[84] = o2.w
      tensor[85] = o2.h
    # important: jump phase, otherwise the model
    # cannot detect that it is in a jump
    tensor[86 + state.jump_phase + 1] = 1
    return tensor

  def call_agent(self, state):
    e = random.random()
    if e < self.epsilon:
      return random.randint(0, dino_env.NUMBER_OF_PLAYER_ACTIONS - 1)
    tensor = self.as_tensor(state)
    predictions = self.online_model(tf.expand_dims(tensor, axis=0))
    action = tf.argmax(predictions[0]).numpy()
    return action

  def train_model(self):
    # Take a batch from replay_buffer and make predictions
    # on this batch. Adapt Q values and re-train with updated
    # values.
    batch = random.sample(self.replay_buffer.buffer, self.batch_size)
    states_batch = np.array([entry[0] for entry in batch])
    next_states_batch = np.array([entry[1] for entry in batch])

    states_preds_online = self.online_model(states_batch).numpy()
    next_states_preds_online = self.online_model(next_states_batch).numpy()
    next_states_preds_target = self.target_model(next_states_batch).numpy()

    # DQN: target model is used to select action and it's prediction
    #      is used as updated value.
    # DDQN: online model is used to select action, but still target 
    #       model's Q value prediction is used as updated value for 
    #       online model.

    # online model is used to select action
    next_actions = tf.argmax(next_states_preds_online, axis=1).numpy()
    for i in range(self.batch_size):
      r = batch[i][2]
      action = batch[i][4]
      terminal = batch[i][3]
      next_action = next_actions[i]

      updated_value = r
      if not terminal:
        next_state_pred_target = next_states_preds_target[i][next_action]

        updated_value += self.gamma * next_state_pred_target      

      # print(f"Values before: {states_preds_online[i]}, updated action: {action}")
      states_preds_online[i][action] = updated_value
      # print(f"Values after: {states_preds_online[i]}")

    self.online_model.fit(
      x = states_batch,
      y = states_preds_online,
      epochs = 1,
      verbose = False
    )

  def play_episode(self, live=True):
    s = dino_env.State()
    done = False
    while not done:
      action = self.call_agent(s)

      next_s, reward, done = s.step(action)

      s_tensor = self.as_tensor(s)
      next_s_tensor = self.as_tensor(next_s)
      self.replay_buffer.append((s_tensor, next_s_tensor, reward, done, action))

      # live?
      if live:
        print(next_s.str())
        print(next_s.score)
        #time.sleep(next_s.speed / 2000) # 2000 = double speed
      
      if len(self.replay_buffer.buffer) > self.batch_size:
        self.train_model()

      s = next_s
    print(f"Episode done, score: {s.score}")

    if self.epsilon > self.epsilon_minimum:
      self.epsilon -= self.epsilon_decrease

  def train(self, episodes):
    for i in range(episodes):
      self.play_episode()

      if i % self.target_update_frequency:
        self.target_model.set_weights(self.online_model.get_weights())
        self.target_model.save('dino.hdf5')


d = DinoTrainer()
d.train(10000)
