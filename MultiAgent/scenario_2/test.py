import convoy
import sys
import torch

from network import FeedForwardActorNN
from eval_policy import eval_policy

def test(env, actor_model):
	"""
		Tests the model.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in
		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = 3
	act_dim = 1

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardActorNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main():
	env = convoy.env(n_vehicles=3)
	test(env=env, actor_model='ppo_actor.pth')

if __name__ == '__main__':
	main()