import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearFeatureBaseline(nn.Module):
	"""
	Linear baseline based on handcrafted features, as described in [1]
	(Supplementary Material 2).

	[1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
		"Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
		(https://arxiv.org/abs/1604.06778)
	"""

	def __init__(self, input_size, reg_coeff=1e-5, is_mlp=True, lr=1e-5, eps=0, is_bounded=False):
		super(LinearFeatureBaseline, self).__init__()
		self.input_size = input_size
		self.hidden_size = 32
		self._reg_coeff = reg_coeff
		self.MLP = is_mlp
		self.epsilon = eps
		self.bounded = is_bounded

		self.build_feature_extractor()
		self.build_optimizer()
		self.build_forward()
		
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

	def build_optimizer(self):
		if self.MLP:
			self.update = self.nnstep
		else:
			self.update = self.fit

	def nnstep(self, episodes):
		_values = self.forward(episodes)
		# _values.requires_grad_()
		values = _values.flatten()

		# Bellman return
		returns = episodes.returns.flatten()

		# TD return
		# returns = episodes.gae(_values).flatten() + values

		loss = F.mse_loss(returns, values)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss

	def fit(self, episodes):
		# sequence_length * batch_size x feature_size
		features = self.feature(episodes).to(self.linear.weight.device)
		values = self.linear(features).flatten()
		loss = F.mse_loss(episodes.returns.flatten(), values)

		featmat = features.view(-1, self.feature_size)
		# sequence_length * batch_size x 1
		returns = episodes.returns.view(-1, 1)

		reg_coeff = self._reg_coeff
		eye = torch.eye(self.feature_size, dtype=torch.float32,
		                device=self.linear.weight.device)
		for _ in range(5):
			try:
				coeffs = torch.linalg.lstsq(
					torch.matmul(featmat.t(), featmat) + reg_coeff * eye,
					torch.matmul(featmat.t(), returns)
				).solution
				break
			except RuntimeError:
				reg_coeff += 10
		else:
			raise RuntimeError('Unable to solve the normal equations in '
			                   '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
			                   'matrix) is not full-rank, regardless of the regularization '
			                   '(maximum regularization: {0}).'.format(reg_coeff))
		self.linear.weight.data = coeffs.data.t()
		return loss

	def build_forward(self):
		if self.bounded == True:
			print("using bounded forward")
			self._forward = self.bounded_forward
		else:
			print("using normal forward")
			self._forward = self.linear
			
	def bounded_forward(self, features):
		l, u = self.compute_bounds(features)
		return l

	def forward(self, episodes):
		
		features = self.feature(episodes)
		values = self._forward(features)
		return values

	def compute_bounds(self, x_bounds):
		l = torch.full_like(x_bounds, -self.epsilon).to(device='cuda')
		u = torch.full_like(x_bounds, self.epsilon).to(device='cuda')
		l += x_bounds
		u += x_bounds
		l = l.view(-1, self.feature_size)
		u = u.view(-1, self.feature_size)
		for layer in self.linear:
			if isinstance(layer, nn.Linear):	
				W, b = layer.weight.t(), layer.bias.t()
				l_out = torch.matmul(l, W.clamp(min=0)) + torch.matmul(u, W.clamp(max=0)) + b
				u_out = torch.matmul(u, W.clamp(min=0)) + torch.matmul(l, W.clamp(max=0)) + b
			else:
				l_out = layer(l)
				u_out = layer(u)
			l = l_out
			u = u_out
		l = l.view(x_bounds.size(0), x_bounds.size(1), 1)
		u = u.view(x_bounds.size(0), x_bounds.size(1), 1)
		return l, u
	
	def build_net(self):
		if self.MLP:
			self.linear = nn.Sequential(
			nn.Linear(self.feature_size, self.hidden_size),
			nn.ReLU(),
			nn.Linear(self.hidden_size, 1)
		)
		else:
			# Duan et al 2016a
			self.linear = nn.Linear(self.feature_size, 1, bias=False)
			self.linear.weight.data.zero_()

	@property
	def feature_size(self):
		return 2 * self.input_size + 4

	def build_feature_extractor(self):
		self.build_net()
		if isinstance(self.linear, nn.Linear):
			print("linearfeature")
			self.feature = self._feature
		elif isinstance(self.linear, nn.Sequential):
			print("mlpfeature")
			if not self.epsilon==0:
				print("adv training")
				if self.bounded:
					print("bounded feature")
					self.feature = self._bounded_feature
				else:
					print("normal feature")
					self.feature = self._feature
			else:
				print("no adv")
				self.feature = self._feature
		else:
			print("not a valid extractor")

	def _feature(self, episodes):
		ones = episodes.mask.unsqueeze(2)
		observations = episodes.observations * ones
		cum_sum = torch.cumsum(ones, dim=0) * ones
		al = cum_sum / 100.0
		return torch.cat([observations, observations ** 2,
		                  al, al ** 2, al ** 3, ones], dim=2)
	
	def _bounded_feature(self, episodes):
		ones = episodes.mask.unsqueeze(2)
		observations = episodes.nopertobs * ones
		cum_sum = torch.cumsum(ones, dim=0) * ones
		al = cum_sum / 100.0
		return torch.cat([observations, observations ** 2,
		                  al, al ** 2, al ** 3, ones], dim=2)
