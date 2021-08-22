import collections
import torch
import torch.nn as nn
import numpy as np
import copy
from helper_functions import ReplayBuffer, map_action_to_rule
from tqdm import tqdm
import time
from itertools import count
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.utils import *
from fastcore.test import *

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    "Transitions", ["state", "action", "reward", "discount", "next_state"]
)
# TODO Try initializing the Q-table by using randomization.
# TODO Try conceptual variations of the replay buffer.
# TODO Try implementing different loss functions.

class Agent():
  def __init__(self,
               env,
               q_network: nn.Module,
               replay_capacity: int = 100_000,
               epsilon: float = 0.01,
               batch_size: int = 10,
               learning_rate: float = 6e-4,
               memory: int = 6,
               discount: float = 0.9):

    # Store all kwargs given to agent (i.e., hyperparameters).
    store_attr()
    # Get size of action space from the environment.
    self._num_actions = env.action_space.n
    # Create a second q net with the same structure and initial values, which
    # we'll be updating separately from the learned q-network.
    self._target_network = copy.deepcopy(self.q_network)
    # Create the replay buffer.
    self._replay_buffer = ReplayBuffer(replay_capacity)
    # Set internal PyTorch parameters.
    self._optimizer = torch.optim.Adam(self.q_network.parameters(),lr = learning_rate)
    self._loss_fn = nn.MSELoss()
    # Setup/reset model state.
    self.reset()
    self._action = np.random.randint(4) # Initialize action (which card was picked)
    self._rule = np.random.randint(4) # Which category was picked on last attempt

  def reset(self):
    """Reset/initialize model parameters, including:
       observation, action, rule, streak and step counters, loss container"""
    self.env.reset()
    self._observation = self.env.card # Get first observation (card) from environment
    self._streak_count = 0 # Track number of successive correct answers

    # Game episode information (see run() implementation)
    self._episode_steps = 0 # Keep an internal tracker of steps
    self._episode_return = 0
    self._episode_loss = 0
    self._last_loss = 0.0 # Container for the computed loss (see run() implementation)
    self._done = False

  def select_action(self, state):
    """Compute Q-values through the prediction network for a given state vector,
       and returns an action following epsilon-greedy policy.

       state: 5-D int vector containing [card, prev_rule, streak_count].
       Returns: action as an integer in range [0,3]."""
    q_values = self.q_values(state)

    if self.epsilon < torch.rand(1):
      action = q_values.argmax(axis=-1)
    else:
      action = torch.randint(low=0, high=self._num_actions , size=(1,), dtype=torch.int64)
    return action

  def q_values(self, state):
    """Computes Q-values through the prediction network for a given state vector.
       Handles both single and batch vectors.
       Returns a tensor."""
    # TODO handle
    print(f"shape of state: {state.shape}")
    # Adds batch dimension.
    q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
    # Removes batch dimension and detaches from graph.
    q_values = q_values.squeeze(0)
    return q_values

  def get_state(self):
    """Returns a 5-D state representation containing [card, prev_rule, streak_count]
       represented as integers."""
    state = [x for x in self._observation] # Unfold card tuple
    state.append(self._rule) # Get previous rule applied
    state.append(self._streak_count)
    return state

  def sample_replay_buffer(self):
    # Sample a minibatch of transitions from experience replay.
    transitions = self._replay_buffer.sample(self.batch_size)

    # Note: each of these tensors will be of shape [batch_size, ...].
    s = torch.FloatTensor(transitions.state)
    a = torch.tensor(transitions.action,dtype=torch.int64)
    r = torch.FloatTensor(transitions.reward)
    d = torch.FloatTensor(transitions.discount)
    next_s = torch.FloatTensor(transitions.next_state)

    return s, a, r, d, next_s

  def next_q_values(self, **kwargs):
    # Compute the Q-values at next states in the transitions.
    with torch.no_grad():
      # NOTE: Not sure if transposing is the canonically "correct" thing to do here.
      q_next_s = self.q_network(next_s.T)  # Shape [batch_size, num_actions].
      max_q_next_s = q_next_s.max(axis=-1)[0]

    return q_next_s, max_q_next_s, kwargs

  def td_error(self, **kwargs):
      # Compute the TD error and then the losses.
      target_q_value = r + d * max_q_next_s
      return target_q_value, kwargs


  def grad_update(self, **kwargs):
    # Compute the Q-values at original state. TODO move this
    q_s = self.q_network(s.T)

    # Gather the Q-value corresponding to each action in the batch.
    q_s_a = q_s.gather(1, a.view(-1,1)).squeeze(1)

    loss = self._loss_fn(target_q_value, q_s_a)

    # Compute the gradients of the loss with respect to the q_network variables.
    self._optimizer.zero_grad()

    loss.backward()
    # Apply the gradient update.
    self._optimizer.step()

    # Store the loss for logging purposes (see run_loop implementation above).
    self._last_loss = loss.detach().numpy()

  def observe_first(self, observation):
    self._replay_buffer.add_first(observation)

  def observe(self, action: int, reward, next_state):
    discount = self.discount ** self._episode_steps
    self._replay_buffer.add(action, reward, next_state, discount)

    return action, reward, next_state, discount

  def update_network(self, **kwargs):
    if not self._replay_buffer.is_ready(self.batch_size):
      # If the replay buffer is not ready to sample from, do nothing.
      return

    pipeline = compose(
      self.sample_replay_buffer,
      self.next_q_values,
      self.td_error,
      self.q_values,
      self.grad_update
    )

    return pipeline, kwargs

  def update_agent(self):
    # Generate an action from the agent's policy and step the environment.
    action = self._action = int(self.select_action(state))
    reward, next_obs, done, _ = self.env.step(action)

    # Update streak counter if reward is positive.
    if reward == 1:
        self._streak_count = min(self._streak_count+1, self.memory)
    else:
        self._streak_count = 0

    # Update internal state.
    self._rule = map_action_to_rule(self._observation, action)
    self._observation = next_obs

    # Get state representation (different from observation)
    next_state = self.get_state()

    return action, reward, next_state

  def log_steps(self, **kwargs):
    self._episode_steps += 1
    self._episode_return += reward

    # if log_loss: # unused for now
    #   self._episode_loss += agent.last_loss

    if logbook:
      logbook.write_actions(episode, self._episode_return)

    return kwargs

  def log_episode(self, logbook, **kwargs):
      if logbook:
        logbook.write_episodes(self._episode, self._episode_steps, self._episode_return)

      self._total_returns.append(self._episode_return)

      return kwargs

  def init_episode(self, episode):
      start_time = time.time() # TODO refactor into logging function
      # Reset parameters and start the environment.
      self.reset()
      # Get first state and store it in the replay buffer.
      state = self.get_state()
      self.observe_first(state)

      self._episode = episode

  # def logfn():

  # @logfn
  def run(self, num_episodes: int = 100,
          log_loss=False, logbook=None,):
    """Perform the run loop (based on Acme implementation).

    Run the environment loop for `num_episodes` episodes. Each episode is itself
    a loop which interacts first with the environment to get an observation and
    then give that observation to the agent in order to retrieve an action. Upon
    termination of an episode a new episode will be started. If the number of
    episodes is not given then this will interact with the environment
    infinitely.

    Args:
      num_episodes: number of episodes to run the loop for. If `None` (default),
        runs without limit.
      log_loss: enable/disable logging results of the loss function.
      logbook: object that handles detailed log data and .csv output.
        If `None` (default), only the list of returns for each episode
        will be saved and subsequently returned by this function.
    """
    iterator = range(num_episodes) if num_episodes else itertools.count()
    self._total_returns = []

    for episode in tqdm(iterator):
      self.init_episode(episode)

      # Run an episode.
      while not self._done:

        pipeline = compose(
          self.update_agent,
          self.observe,
          self.update_network,
          self.log_steps
        )

        result = pipeline()

      self.log_episode(logbook)

    return all_returns
