import numpy as np
import torch
from event_utils import *
from abc import ABC, abstractmethod

class warp_function(ABC):
"""
Base class for objects that can warp events to a reference time
via a parametrizeable, differentiable motion model
"""
    def __init__(self, name, dims):
        """
        Constructor.
        @param name The name of the warp function (eg 'optic flow')
        @param dims The number of degrees of freedom of the motion model
        """
        self.name = name
        self.dims = dims
        super().__init__()

    @abstractmethod
    def warp(self, xs, ys, ts, ps, t0, params, compute_grad=False):
        """
        Warp function which given a set of events and a reference time,
        moves the events to that reference time via a motion model
        @param xs x components of events as list
        @param ys y components of events as list
        @param ts t components of events as list
        @param ps p components of events as list
        @param t0 The reference time to which to warp the events to
        @param params The parameters of the motion model for
            which to warp the events
        @param compute_grad If True, compute the gradient of the warp with 
            respect to the motion parameters for each event (the Jacobian)
        @returns xs_warped, ys_warped, xs_jacobian, ys_jacobian: The warped
            event locations and the gradients for each event as a tuple of four
            numpy arrays
        """
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

class xyztheta_warp(warp_function):
    """
    This class implements 4-DoF x,y,z,rotation warps from Mitrokhin etal, 
    "Event-based moving object detection and tracking"
    """
    def __init__(self):
        warp_function.__init__(self, 'xyztheta_warp', 4)

    def warp(self, xs, ys, ts, ps, t0, params, compute_grad=False):
        pass

class pure_rotation_warp(warp_function):
    """
    This class implements pure rotation warps, with params
    x,y,theta (x,y is center of rotation, theta is angular velocity
    """
    def __init__(self):
        warp_function.__init__(self, 'pure_rotation_warp', 4)
{not:timeslice}
    def warp(self, xs, ys, ts, ps, t0, params, compute_grad=False):
        pass
