import numpy as np
import torch
from ..util.event_util import *
from ..representations.image import *
from scipy.ndimage.filters import gaussian_filter
from abc import ABC, abstractmethod

class objective_function(ABC):

    def __init__(self, name="template", use_polarity=True,
            has_derivative=True, default_blur=1.0):
        self.name = name
        self.use_polarity = use_polarity
        self.has_derivative = has_derivative
        self.default_blur = default_blur
        super().__init__()

    @abstractmethod
    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Define the warp function. Either give the params and the events or give a
        precomputed iwe (if xs, ys, ts, ps are given, iwe is not necessary).
        """
        #if iwe is None:
        #    iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size,
        #            use_polarity=self.use_polarity, compute_gradient=False)
        #blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        #if blur_sigma > 0:
        #    iwe = gaussian_filter(iwe, blur_sigma)
        #loss = compute_loss_here...
        #return loss
        pass

    @abstractmethod
    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        If your warp function has it, define the gradient (otherwise set has_derivative to False
        and numeric grads will be used). Either give the params and the events or give a
        precomputed iwe and d_iwe (if xs, ys, ts, ps are given, iwe, d_iwe is not necessary).
        """
        #if iwe is None or d_iwe is None:
        #    iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size,
        #            use_polarity=self.use_polarity, compute_gradient=True)
        #blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        #if blur_sigma > 0:
        #    d_iwe = gaussian_filter(d_iwe, blur_sigma)

        #gradient = []
        #for grad_dim in range(d_iwe.shape[0]):
        #    gradient.append(compute_gradient_here...)
        #grad = np.array(gradient)
        #return grad
        pass

def get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, compute_gradient=False, use_polarity=True):
    """
    Given a set of parameters, events and warp function, get the warped image and derivative image
    if required.
    """
    if not use_polarity:
        ps = np.abs(ps)
    xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ps, ts[-1], params, compute_grad=compute_gradient)
    mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0])
    xs, ys, ts, ps = xs*mask, ys*mask, ts*mask, ps*mask
    if compute_gradient:
        jx, jy = jx*mask, jy*mask
    iwe, iwe_drv = events_to_image_drv(xs, ys, ps, jx, jy,
            interpolation='bilinear', compute_gradient=compute_gradient)
    return iwe, iwe_drv


class variance_objective(objective_function):
    """
    Variance objective from 'Gallego, Accurate Angular Velocity Estimation with an Event Camera, RAL'17'
    """
    def __init__(self):
        self.use_polarity = True
        self.name = "variance"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by var(g(x)) where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        loss = np.var(iwe-np.mean(iwe))
        return -loss

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*(g(x)-mu(g(x))*(g'(x)-mu(g'(x))) where g(x) is the IWE
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        zero_mean = 2.0*(iwe-np.mean(iwe))
        for grad_dim in range(d_iwe.shape[0]):
            mean_jac = d_iwe[grad_dim]-np.mean(d_iwe[grad_dim])
            gradient.append(np.mean(zero_mean*(d_iwe[grad_dim]-np.mean(d_iwe[grad_dim]))))
        grad = np.array(gradient)
        return -grad

class rms_objective(objective_function):
    """
    Root mean squared objective
    """
    def __init__(self):
        self.use_polarity = True
        self.name = "rms"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by l2(g(x))^2 where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        norm = np.linalg.norm(iwe, 2)
        num_pix = iwe.shape[0]*iwe.shape[1]
        loss = (norm*norm)/num_pix
        return -loss

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*(mu(g(x)*g'(x))) where g(x) is IWE
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        for grad_dim in range(d_iwe.shape[0]):
            gradient.append(2.0*np.mean(iwe*d_iwe[grad_dim]))
        grad = np.array(gradient)
        return -grad

class sos_objective(objective_function):
    """
    Sum of squares objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """

    def __init__(self):
        self.use_polarity = True
        self.name = "sos"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by g(x)^2 where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        sos = np.mean(iwe*iwe)#/num_pix
        return -sos

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*g(x)*g'(x) where g(x) is IWE
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        img_component = 2.0*iwe
        for grad_dim in range(d_iwe.shape[0]):
            gradient.append(np.mean(d_iwe[grad_dim]*img_component))
        grad = np.array(gradient)
        return -grad

class soe_objective(objective_function):
    """
    Sum of exponentials objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self):
        self.use_polarity = False
        self.name = "soe"
        self.has_derivative = True
        self.default_blur=2.5

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by e^g(x) where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        exp = np.exp(iwe.astype(np.double))
        soe = np.mean(exp)
        return -soe

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by e^g(x)*g'(x) where g(x) is IWE
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)
            iwe = gaussian_filter(iwe, blur_sigma)
        gradient = []
        soe_deriv = np.exp(iwe.astype(np.double))#/num_pix
        for grad_dim in range(d_iwe.shape[0]):
            gradient.append(np.mean(soe_deriv*d_iwe[grad_dim]))
        grad = np.array(gradient)
        return -grad

class moa_objective(objective_function):
    """
    Max of accumulations objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self):
        self.use_polarity = False
        self.name = "moa"
        self.has_derivative = False
        self.default_blur=3.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by max(g(x)) where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        moa = np.max(iwe)
        return -moa

    def evaluate_gradient(self, iwe=None, d_iwe=None, blur_sigma=None, showimg=False):
        """
        No analytic derivative known
        """
        return None

class isoa_objective(objective_function):
    """
    Inverse sum of accumulations objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self, thresh=0.5):
        self.use_polarity = False
        self.thresh = thresh
        self.name = "isoa"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by sum(1 where g(x)>1 else 0) where g(x) is IWE.
        This formulation has similar properties to original ISoA, but negation makes derivative
        more stable than inversion.
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        isoa = np.sum(np.where(iwe>self.thresh, 1, 0))
        return isoa

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient = g'(x) where thresh<g(x), otherwise 0
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
            d_iwe = gaussian_filter(d_iwe, blur_sigma)
        gradient = []
        mask = np.ma.masked_greater(iwe, self.thresh)
        iwe[iwe > self.thresh] = 1.0
        iwe[iwe <= self.thresh] = 0.0
        for grad_dim in range(d_iwe.shape[0]):
            gradient.append(np.sum(d_iwe[grad_dim]*iwe))
        grad = np.array(gradient)
        return -grad

class sosa_objective(objective_function):
    """
    Sum of Supressed Accumulations objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self, p=3):
        self.p = p
        self.use_polarity = False
        self.name = "sosa"
        self.has_derivative = True
        self.default_blur=2.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by e^(-p*g(x)) where g(x) is IWE. p is arbitrary shifting factor,
        higher values give better noise performance but lower accuracy.
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        exp = np.exp(-self.p*iwe.astype(np.double))
        sosa = np.sum(exp)
        return -sosa

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient = p*-e^(-p*g(x))*g'(x) where g(x) is iwe
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
            d_iwe = gaussian_filter(d_iwe, blur_sigma)
        gradient = []
        exp = np.exp((-self.p*iwe).astype(np.double))
        fx = -self.p*exp
        for grad_dim in range(d_iwe.shape[0]):
            gradient.append(np.sum(d_iwe[grad_dim]*fx))
        grad = np.array(gradient)
        return -grad

class zhu_timestamp_objective(objective_function):
    """
    Squared timestamp images objective (Zhu et al, Unsupervised Event-based
    Learning of Optical Flow, Depth, and Egomotion, CVPR19)
    """
    def __init__(self):
        self.use_polarity = True
        self.name = "zhu"
        self.has_derivative = False
        self.default_blur=2.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by g(x)^2*h(x)^2 where g(x) is image of average timestamps of positive events
        and h(x) is image of average timestamps of negative events.
        """
        if iwe is None:
            xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ps, ts[-1], params, compute_grad=False)
            mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0])
            xs, ys, ts, ps = xs*mask, ys*mask, ts*mask, ps*mask
            posimg, negimg = events_to_zhu_timestamp_image(xs, ys, ts, ps, compute_gradient=False, showimg=showimg)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            posimg = gaussian_filter(posimg, blur_sigma)
            negimg = gaussian_filter(negimg, blur_sigma)
        loss = -(np.sum(posimg*posimg)+np.sum(negimg*negimg))
        return loss

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        No derivative known
        """
        return None

class r1_objective(objective_function):
    """
    R1 objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self, p=3):
        self.name = "r1"
        self.use_polarity = False
        self.has_derivative = False
        self.p = p
        self.default_blur = 1.0
        self.last_sosa = 0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by SOS and SOSA combined
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        sos = np.mean(iwe*iwe)
        exp = np.exp(-self.p*iwe.astype(np.double))
        sosa = np.sum(exp)
        if sosa > self.last_sosa:
            return -sos
        self.last_sosa = sosa
        return -sos*sosa

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        No derivative known
        """
        return None
