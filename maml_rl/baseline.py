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

	def __init__(self, input_size, reg_coeff=1e-5, is_mlp=True, lr=1e-5):
		super(LinearFeatureBaseline, self).__init__()
		self.input_size = input_size
		self.hidden_size = 32
		self._reg_coeff = reg_coeff
		self.MLP = is_mlp
		self.build_feature_extractor()
		self.build_optimizer()
		self.epsilon = 0.1
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

	def forward(self, episodes):
		# IBP Lower Bound
		# =================================
		# out = torch.empty((0,))
		# out_ep = torch.empty((0,))
		# for feature_ep in features:
		# 	for feature_worker in feature_ep:
		# 		l, u =  self.compute_bounds(feature_worker, layer=self.linear)
		# 		out_ep = torch.cat((out_ep, l.unsqueeze(0)))
		# 	out = torch.cat((out, out_ep.unsqueeze(0)))
		# 	out_ep = torch.empty((0,))
		# value = out
		# =================================

		features = self.feature(episodes)
		values = self.linear(features)
		return values

	def compute_bounds(self, x_bounds, layer):
		l = torch.full_like(x_bounds, -self.epsilon)
		u = torch.full_like(x_bounds, self.epsilon)
		l += x_bounds
		u += x_bounds
		# l.to(torch.device('cuda'))
		# u.to(torch.device('cuda'))
		W, b = layer.weight, layer.bias
		# l_out = torch.matmul(W.clamp(min=0), l) + torch.matmul(W.clamp(max=0), u) + b
		# u_out = torch.matmul(W.clamp(min=0), u) + torch.matmul(W.clamp(max=0), l) + b
		l_out = torch.matmul(W.clamp(min=0), l) + torch.matmul(W.clamp(max=0), u)
		u_out = torch.matmul(W.clamp(min=0), u) + torch.matmul(W.clamp(max=0), l)
		return l_out, u_out
	
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
			self.feature = self._feature
		else:
			print("not a valid extractor")

	def _feature(self, episodes):
		ones = episodes.mask.unsqueeze(2)
		observations = episodes.observations * ones
		# observations = episodes.nopertobs * ones
		cum_sum = torch.cumsum(ones, dim=0) * ones
		al = cum_sum / 100.0
		return torch.cat([observations, observations ** 2,
		                  al, al ** 2, al ** 3, ones], dim=2)
	
	def _MLPfeature(self, episodes):
		ones = episodes.mask.unsqueeze(2)
		observations = episodes.observations * ones
		# observations = episodes.nopertobs * ones
		return observations