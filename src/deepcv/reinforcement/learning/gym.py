#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" GYM Reinforcement learning module - gym.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import logging

import gym
import torch
import torch.nn
import numpy as np

import deepcv.utils

__all__ = ['random_agent', 'run_environment']

def random_agent(observation : Union[np.ndarray, torch.Tensor], action_space : gym.Space, observation_space: gym.Space):
    return action_space.sample()

def run_environment(agent: Callable[[gym.Env, gym.Space, gym.Space], Union[deepcv.utils.Number, np.ndarray]], environment: gym.Env, episodes: int, max_actions: int = 1e4)
    for i_episode in range(episodes):
        observation = environment.reset()
        for t in range(max_actions):
            environment.render()
            print(observation)
            action = agent(observation, environment.action_space, environment.observation_space)
            observation, reward, done, info = environment.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    run_environment(random_agent, env, episodes=10, max_actions=1e3)
    env.close()

# TODO: remove this example code from gym
# import os
# import sys
# import json

# import pickle
# import argparse
# import numpy as np

# import gym
# from policies import BinaryActionLinearPolicy

# def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
#     """
#     Generic implementation of the cross-entropy method for maximizing a black-box function
#     Args:
#         f: a function mapping from vector -> scalar
#         th_mean (np.array): initial mean over input distribution
#         batch_size (int): number of samples of theta to evaluate per batch
#         n_iter (int): number of batches
#         elite_frac (float): each batch, select this fraction of the top-performing samples
#         initial_std (float): initial standard deviation over parameter vectors
#     returns:
#         A generator of dicts. Subsequent dicts correspond to iterations of CEM algorithm.
#         The dicts contain the following values:
#         'ys' :  numpy array with values of function evaluated at current population
#         'ys_mean': mean value of function over current population
#         'theta_mean': mean value of the parameter vector over current population
#     NOTE: Code from https://github.com/openai/gym/blob/master/examples/agents/cem.py
#     """
#     n_elite = int(np.round(batch_size*elite_frac))
#     th_std = np.ones_like(th_mean) * initial_std

#     for _ in range(n_iter):
#         ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
#         ys = np.array([f(th) for th in ths])
#         elite_inds = ys.argsort()[::-1][:n_elite]
#         elite_ths = ths[elite_inds]
#         th_mean = elite_ths.mean(axis=0)
#         th_std = elite_ths.std(axis=0)
#         yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

# def do_rollout(agent, env, num_steps, render=False):
#     """ NOTE: Code from https://github.com/openai/gym/blob/master/examples/agents/cem.py """
#     total_rew = 0
#     ob = env.reset()
#     for t in range(num_steps):
#         a = agent.act(ob)
#         (ob, reward, done, _info) = env.step(a)
#         total_rew += reward
#         if render and t%3 == 0: env.render()
#         if done: break
#     return total_rew, t+1

# if __name__ == '__main__':
#     # NOTE: Code from https://github.com/openai/gym/blob/master/examples/agents/cem.py
#     gym.logger.set_level(gym.logger.INFO)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--display', action='store_true')
#     parser.add_argument('target', nargs="?", default="CartPole-v0")
#     args = parser.parse_args()

#     env = gym.make(args.target)
#     env.seed(0)
#     np.random.seed(0)
#     params = dict(n_iter=10, batch_size=25, elite_frac=0.2)
#     num_steps = 200

#     # You provide the directory to write to (can be an existing
#     # directory, but can't contain previous monitor results. You can
#     # also dump to a tempdir if you'd like: tempfile.mkdtemp().
#     outdir = '/tmp/cem-agent-results'
#     env = gym.warppers.Monitor(env, outdir, force=True)

#     # Prepare snapshotting
#     # ----------------------------------------
#     def writefile(fname, s):
#         with open(os.path.join(outdir, fname), 'w') as fh: fh.write(s)
        
#     info = {'params': params, 'argv': sys.argv, 'env_id': env.spec.id}
#     # ------------------------------------------

#     def noisy_evaluation(theta):
#         agent = BinaryActionLinearPolicy(theta)
#         rew, T = do_rollout(agent, env, num_steps)
#         return rew

#     # Train the agent, and snapshot each stage
#     for (i, iterdata) in enumerate(
#         cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
#         print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
#         agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
#         if args.display: do_rollout(agent, env, 200, render=True)
#         writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))

#     # Write out the env at the end so we store the parameters of this
#     # environment.
#     writefile('info.json', json.dumps(info))

#     env.close()