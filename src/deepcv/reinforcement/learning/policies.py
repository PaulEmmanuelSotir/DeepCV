#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Reinforcement learning policies module - policies.py - `DeepCV`__  
File gathering various policy models for reinforcement learning problems.
.. moduleauthor:: Paul-Emmanuel Sotir
"""

__all__ = ['BinaryActionLinearPolicy', 'ContinuousActionLinearPolicy']


class BinaryActionLinearPolicy(object):
    """ Code from https://github.com/openai/gym/blob/master/examples/agents/_policies.py """

    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a


class ContinuousActionLinearPolicy(object):
    """ Code from https://github.com/openai/gym/blob/master/examples/agents/_policies.py """

    def __init__(self, theta, n_in, n_out):
        assert len(theta) == (n_in + 1) * n_out
        self.W = theta[0: n_in * n_out].reshape(n_in, n_out)
        self.b = theta[n_in * n_out: None].reshape(1, n_out)

    def act(self, ob):
        a = ob.dot(self.W) + self.b
        return a
