import numpy as np
import torch
from event_utils import *
from abc import ABC, abstractmethod

class warp_function(ABC):

    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
        super().__init__()

    @abstractmethod
    def warp(self, xs, ys, ts, ps, t0, params, compute_grad=False):
        #Warp the events...
        #if compute_grad:
        #   compute the jacobian of the warp function
        pass

class linvel_warp(warp_function):
    """
    This class implements linear velocity warping (global optic flow)
    """
    def __init__(self):
        warp_function.__init__(self, 'linvel_warp', 2)

    def warp(self, xs, ys, ts, ps, t0, params, compute_grad=False):
        dt = ts-t0
        x_prime = xs-dt*params[0]
        y_prime = ys-dt*params[1]
        jacobian_x, jacobian_y = None, None
        if compute_grad:
            jacobian_x = np.zeros((2, len(x_prime)))
            jacobian_y = np.zeros((2, len(y_prime)))
            jacobian_x[0, :] = -dt
            jacobian_y[1, :] = -dt
        return x_prime, y_prime, jacobian_x, jacobian_y
