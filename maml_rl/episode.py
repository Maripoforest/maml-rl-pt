import numpy as np
import torch
import torch.nn.functional as F


class BatchEpisodes:

	def __init__(self, batch_size, gamma=0.95, device='cpu'):
		self.batch_size = batch_size
		self.gamma = gamma
		self.device = device

		# [[], [],...batchsz of []]
		self._observations_list = [[] for _ in range(batch_size)]
		self._actions_list = [[] for _ in range(batch_size)]
		self._rewards_list = [[] for _ in range(batch_size)]
		self._nopertobs_list = [[] for _ in range(batch_size)]

		self._mask_list = []

		self._observations = None
		self._actions = None
		self._rewards = None
		self._returns = None
		self._nopertobs = None
		self._mask = None

	@property
	def observations(self):
		if self._observations is None:
			observation_shape = self._observations_list[0][0].shape
			observations = np.zeros((len(self), self.batch_size)
			                        + observation_shape, dtype=np.float32)
			for i in range(self.batch_size):
				length = len(self._observations_list[i])
				observations[:length, i] = np.stack(self._observations_list[i], axis=0)
			self._observations = torch.from_numpy(observations).to(self.device)
		return self._observations

	@property
	def nopertobs(self):
		if self._nopertobs is None:
			nopertobs_shape = self._nopertobs_list[0][0].shape
			nopertobs = np.zeros((len(self), self.batch_size)
			                        + nopertobs_shape, dtype=np.float32)
			for i in range(self.batch_size):
				length = len(self._nopertobs_list[i])
				nopertobs[:length, i] = np.stack(self._nopertobs_list[i], axis=0)
			self._nopertobs = torch.from_numpy(nopertobs).to(self.device)
		return self._nopertobs

	@property
	def actions(self):
		if self._actions is None:
			action_shape = self._actions_list[0][0].shape
			actions = np.zeros((len(self), self.batch_size)
			                   + action_shape, dtype=np.float32)
			for i in range(self.batch_size):
				length = len(self._actions_list[i])
				actions[:length, i] = np.stack(self._actions_list[i], axis=0)
			self._actions = torch.from_numpy(actions).to(self.device)
		return self._actions

	@property
	def rewards(self):
		if self._rewards is None:
			rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
			for i in range(self.batch_size):
				length = len(self._rewards_list[i])
				rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
			self._rewards = torch.from_numpy(rewards).to(self.device)
		return self._rewards

	@property
	def returns(self):
		if self._returns is None:
			return_ = np.zeros(self.batch_size, dtype=np.float32)
			returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
			rewards = self.rewards.cpu().numpy()
			mask = self.mask.cpu().numpy()
			for i in range(len(self) - 1, -1, -1):
				return_ = self.gamma * return_ + rewards[i] * mask[i]
				returns[i] = return_
			self._returns = torch.from_numpy(returns).to(self.device)
		return self._returns

	@property
	def mask(self):
		if self._mask is None:
			mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
			for i in range(self.batch_size):
				length = len(self._actions_list[i])
				mask[:length, i] = 1.0
			self._mask = torch.from_numpy(mask).to(self.device)
		return self._mask

	def gae(self, values, tau=1.0):
		"""

		:param values: [200, 20, 1], tensor
		:param tau:
		:return:
		"""
		# Add an additional 0 at the end of values for
		# the estimation at the end of the episode
		values = values.squeeze(2).detach() # [200, 20]
		values = F.pad(values * self.mask, (0, 0, 0, 1)) # [201, 20]

		deltas = self.rewards + self.gamma * values[1:] - values[:-1] # [200, 20]
		advantages = torch.zeros_like(deltas).float() # [200, 20]
		gae = torch.zeros_like(deltas[0]).float() # [20]
		for i in range(len(self) - 1, -1, -1):
			gae = gae * self.gamma * tau + deltas[i]
			advantages[i] = gae

		return advantages

	def append(self, observations, nopertobs, actions, rewards, batch_ids):
		for observation, nopertobs, action, reward, batch_id in zip(observations, nopertobs, actions, rewards, batch_ids):
			if batch_id is None:
				continue
			self._observations_list[batch_id].append(observation.astype(np.float32))
			self._actions_list[batch_id].append(action.astype(np.float32))
			self._rewards_list[batch_id].append(reward.astype(np.float32))
			self._nopertobs_list[batch_id].append(nopertobs.astype(np.float32))

	def __len__(self):
		return max(map(len, self._rewards_list))