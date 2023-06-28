import  gym
import  torch
import  multiprocessing as mp
import numpy as np

from    maml_rl.envs.subproc_vec_env import SubprocVecEnv
from    maml_rl.episode import BatchEpisodes


def make_env(env_name):
	"""
	return a function
	:param env_name:
	:return:
	"""
	def _make_env():
		return gym.make(env_name)

	return _make_env


class BatchSampler:

	def __init__(self, env_name, batch_size, num_workers=mp.cpu_count(), epsilon=0):
		"""

		:param env_name:
		:param batch_size: fast batch size
		:param num_workers:
		"""
		self.env_name = env_name
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.epsilon = epsilon
		self.build_sampler()

		self.queue = mp.Queue()
		# [lambda function]
		env_factorys = [make_env(env_name) for _ in range(num_workers)]
		# this is the main process manager, and it will be in charge of num_workers sub-processes interacting with
		# environment.
		self.envs = SubprocVecEnv(env_factorys, queue_=self.queue)
		self._env = gym.make(env_name)

	def build_sampler(self):
		if self.epsilon == 0:
			self.sample = self.sample_without_per
			print("using normal training")

		else:
			self.sample = self.sample_with_per
			print("using adv training")

	def sample_without_per(self, policy, params=None, gamma=0.95, device='cpu'):
		"""

		:param policy:
		:param params:
		:param gamma:
		:param device:
		:return:
		"""
		episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
		for i in range(self.batch_size):
			self.queue.put(i)
		for _ in range(self.num_workers):
			self.queue.put(None)

		observations, batch_ids = self.envs.reset()
		dones = [False]
		while (not all(dones)) or (not self.queue.empty()): # if all done and queue is empty
			# for reinforcement learning, the forward process requires no-gradient
			with torch.no_grad():
				# convert observation to cuda
				# compute policy on cuda
				# convert action to cpu

				nopertobs = observations

				# Perturbed model
				# =======================
				# perturbations = np.random.uniform(-0.1, 0.1, observations.shape)
				# observations += perturbations
				# =======================
				observations_tensor = torch.from_numpy(observations).to(device=device)
				
				# forward via policy network
				# policy network will return Categorical(logits=logits)
				actions_tensor = policy(observations_tensor, params=params).sample()
				actions = actions_tensor.cpu().numpy()

			new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
			print(rewards.shape)
			# here is observations NOT new_observations, batch_ids NOT new_batch_ids
			episodes.append(observations, nopertobs, actions, rewards, batch_ids)
			observations, batch_ids = new_observations, new_batch_ids

		return episodes
	
	def sample_with_per(self, policy, params=None, gamma=0.95, device='cpu'):
		"""

		:param policy:
		:param params:
		:param gamma:
		:param device:
		:return:
		"""
		episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
		for i in range(self.batch_size):
			self.queue.put(i)
		for _ in range(self.num_workers):
			self.queue.put(None)

		observations, batch_ids = self.envs.reset()
		dones = [False]
		while (not all(dones)) or (not self.queue.empty()): # if all done and queue is empty
			# for reinforcement learning, the forward process requires no-gradient
			with torch.no_grad():
				# convert observation to cuda
				# compute policy on cuda
				# convert action to cpu

				nopertobs = observations

				# Perturbed model
				# =======================
				# perturbations = np.random.uniform(-self.epsilon, self.epsilon, observations.shape)
				# print(observations.shape)
				# print(type(observations[0][0]))
				perturbations = np.random.choice([1-self.epsilon, 1+self.epsilon], p=[0.5, 0.5], size=observations.shape)
				observations = np.multiply(perturbations, observations).astype(np.float32)
				# print(observations)
				# print(type(observations[0][0]))
				# =======================
				
				observations_tensor = torch.from_numpy(observations).to(device=device)
				# forward via policy network
				# policy network will return Categorical(logits=logits)
				actions_tensor = policy(observations_tensor, params=params).sample()
				actions = actions_tensor.cpu().numpy()

			new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
			# here is observations NOT new_observations, batch_ids NOT new_batch_ids
			episodes.append(observations, nopertobs, actions, rewards, batch_ids)
			observations, batch_ids = new_observations, new_batch_ids

		return episodes

	def reset_task(self, task):
		tasks = [task for _ in range(self.num_workers)]
		reset = self.envs.reset_task(tasks)
		return all(reset)

	def sample_tasks(self, _tasks):
		tasks = self._env.unwrapped.sample_tasks(_tasks)
		return tasks