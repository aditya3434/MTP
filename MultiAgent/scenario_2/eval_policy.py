import torch
from torch.distributions import MultivariateNormal
import numpy as np

cov_var = torch.full(size=(1,), fill_value=0.8)
cov_mat = torch.diag(cov_var)

def get_action(policy, obs):
	"""
		Queries an action from the actor network, should be called from rollout.

		Parameters:
			obs - the observation at the current timestep

		Return:
			action - the action to take, as a numpy array
			log_prob - the log probability of the selected action in the distribution
	"""
	# Query the actor network for a mean action
	mean = policy(obs)

	# Create a distribution with the mean action and std from the covariance matrix above.
	# For more information on how this distribution works, check out Andrew Ng's lecture on it:
	# https://www.youtube.com/watch?v=JjB58InuTqM
	dist = MultivariateNormal(mean, cov_mat)
	dist_new = dist

	# Sample an action from the distribution
	action = dist.sample()

	# Calculate the log probability for that action
	log_prob = dist.log_prob(action)

	# Return the sampled action and the log probability of that action in our distribution
	return action.detach().numpy(), mean.detach().numpy()

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.
			Parameters:
				None
			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 
		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.
		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not env.done:
			t += 1

			rew = 0

			# Render environment if specified, off by default
			if render:
				env.render()

			for agent in env.agent_iter(3):
				obs, _rew, _done, info = env.last()
				
				if obs.any() == None:
					env.step([0.0])
					continue

				action, log_prob = get_action(policy, obs)
				
				env.step(action)

				rew += _rew

				if (env.done):
					done = True
					break

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(policy, env, render=True):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 
		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.
		Return:
			None
		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)