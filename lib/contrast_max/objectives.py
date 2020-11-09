import numpy as np
import torch
from ..util.event_util import *
from ..representations.image import *
from scipy.ndimage.filters import gaussian_filter
from abc import ABC, abstractmethod
from ..util.util import plot_image
import cv2 as cv

class objective_function(ABC):
"""
Parent class for all objective functions for contrast maximisation
"""
    def __init__(self, name="template", use_polarity=True,
            has_derivative=True, default_blur=1.0, adaptive_lifespan=False,
            pixel_crossings=5, minimum_events=10000):
        """
        Constructor, sets member variables.
        @param name Sets the name of the objective function (eg: 'variance')
        @param use_polarity If true, use the polarity of the events in generating IWEs 
        @param has_derivative If true, this function has a defined analytical derivative.
            Else, will use numerically estimated derivatives.
        @param default_blur Sets the default standard deviation for the Gaussian blurring kernel
        @param adaptive_lifespan Many implementations of contrast maximisation use assumptions of
            linear motion wrt the chosen motion model. A given estimate of the motion parameters
            implies a lifespan of the events. If 'adaptive_lifespan' is True, the number of events
            used during warping is cut to that lifespan for each optimisation step, computed using
            'pixel_crossings'. EG If motion model is optic flow velocity and the
            estimate = 12 pixels/second and 'pixel_crossings'=3, then the lifespan will
            be 3/12=0.25 seconds.
        @param pixel_crossings Number of pixel crossings used to calculate 'adaptive_lifespan'
        @param minimum_events The minimal number of events that 'adaptive_lifespan' will cut to
        """
        self.name = name
        self.use_polarity = use_polarity
        self.has_derivative = has_derivative
        self.default_blur = default_blur
        self.adaptive_lifespan = adaptive_lifespan
        self.pixel_crossings = pixel_crossings
        self.minimum_events = minimum_events

        self.recompute_lifespan = True
        self.lifespan = 0.5
        self.s_idx = 0
        self.num_events = None
        super().__init__()

    @abstractmethod
    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Define the warp function. The function can either receive events and motion
        paramters as input, to compute the IWE and evaluate the objective function,
        or receive a precomputed IWE. An example is given in comments.
        @param params The motion parameters to evaluate at
        @param xs x components of events as list
        @param ys y components of events as list
        @param ts t components of events as list
        @param ps p components of events as list
        @param warpfunc The desired warping function
        @param img_size The size of the image sensor/resolution
        @param blur_sigma The desired amount of blurring to apply to IWE
        @param show_img Debugging tool, if true, show the IWE in a matplotlib window
        @param iwe Precomputed IWE to evalute the objective function for
        @returns Evaluation of objective function at parameters 'params'
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
        Define the gradient of the warp function, if available (else numeric gradient
        will be computed). The function can either receive events and motion
        paramters as input, to compute the IWE and dIWE/dParams and evaluate the objective function,
        or receive a precomputed IWE and dIWE/dParams. An example is given in comments.
        @param params The motion parameters to evaluate at
        @param xs x components of events as list
        @param ys y components of events as list
        @param ts t components of events as list
        @param ps p components of events as list
        @param warpfunc The desired warping function
        @param img_size The size of the image sensor/resolution
        @param blur_sigma The desired amount of blurring to apply to IWE
        @param show_img Debugging tool, if true, show the IWE in a matplotlib window
        @param iwe Precomputed IWE to evalute the objective function for
        @param iwe Precomputed gradient of IWE wrt motion params to evalute the gradient
            of the objective function
        @returns Gradient of objective function wrt motion parameters at 'params'
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

    def iter_update(self, params, pixel_crossings=None):
        """
        Housekeeping function that runs as a callback at each optimisation step
        if 'adaptive_lifespan' is set True
        @param The current motion parameters
        @param The number of pixel crossings to compute the new lifespan
        """
        pixel_crossings = self.pixel_crossings if pixel_crossings is None else pixel_crossings
        magnitude = np.linalg.norm(params)
        if magnitude == 0:
            dt = 5
        else:
            dt = pixel_crossings/magnitude
        self.lifespan = dt
        self.recompute_lifespan = True

    def update_lifespan(self, ts):
        """
        Set the new lifespan and thus the new set of events to be used in optimisation
        @param ts The timestamps of the events currently used
        """
        print("update lifespan")
        if self.adaptive_lifespan:
            self.s_idx = np.searchsorted(ts, ts[-1]-self.lifespan)
            self.s_idx = len(ts)-self.minimum_events if len(ts)-self.s_idx < self.minimum_events else self.s_idx
            print("New num events = {}/{}".format(len(ts)-self.s_idx, len(ts)))
        if self.num_events is None:
            self.num_events = len(ts)-self.s_idx


def cut_events_to_lifespan(xs, ys, ts, ps, params, pixel_crossings, minimum_events=10000):
    """
    Given events, cut the events down to the lifespan defined by the motion parameters
    and desired pixel crossings
    @param xs x components of events as list
    @param ys y components of events as list
    @param ts t components of events as list
    @param ps p components of events as list
    @param params The motion parameters to evaluate at
    @param pixel_crossings Number of pixel crossings used to calculate new lifespan
    @param minimum_events The minimal number of events that the output set of
        events will contain
    @returns The set of events cut to the new lifespan*desired pixel crossings
    """
    magnitude = np.linalg.norm(params)
    dt = pixel_crossings/magnitude
    s_idx = np.searchsorted(ts, ts[-1]-dt)
    num_events = len(xs)-s_idx
    s_idx = len(xs)-minimum_events if num_events < minimum_events else s_idx
    print("Magnitude: {:.2f} pix/s. dt({:.2f} pix)={}. New range is {}:{}={} events".format(magnitude, pixel_crossings, dt, s_idx, len(xs), len(xs)-s_idx))
    return xs[s_idx:-1], ys[s_idx:-1], ts[s_idx:-1], ps[s_idx:-1]

def get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, compute_gradient=False,
        use_polarity=True, return_events=False, return_per_event_contrast=False):
    """
    Given a set of parameters, events and warp function, get the warped image and derivative image
    if required.
    @param params The motion parameters to evaluate at
    @param xs x components of events as list
    @param ys y components of events as list
    @param ts t components of events as list
    @param ps p components of events as list
    @param warpfunc The desired warping function
    @param img_size The size of the image sensor/resolution
    @param compute_gradient If True, compute and return the gradient of the IWE wrt motion params
    @param use_polarity If True, use the polarity of the events in IWE formation
    @param return_events If True, return the warped events as well
    @param return_per_event_contrast If True, return the contrast in the IWE at
        each warped event's location
    @returns IWE, dIWE/dParams, warped events, local contrast of each event in IWE
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
    returnval = [iwe, iwe_drv]
    if return_events:
        returnval.append((xs, ys))
    if return_per_event_contrast:
        weights = image_to_event_weights(xs, ys, iwe)
        returnval.append(weights)
    return tuple(returnval)


class variance_objective(objective_function):
    """
    Variance objective from 'Gallego, Accurate Angular Velocity Estimation with an Event Camera, RAL'17'
    """
    def __init__(self, adaptive_lifespan=False, minimum_events=10000):
        super().__init__(name="variance", use_polarity=True, has_derivative=True,
                default_blur=1.0, adaptive_lifespan=adaptive_lifespan, pixel_crossings=5,
                minimum_events=minimum_events)

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by var(g(x)) where g(x) is IWE
        """
        if iwe is None:
            if self.adaptive_lifespan:
                #print("{}/{}".format(self.s_idx, len(ts)))
                #ps = ps/len(ps)*100000
                if self.recompute_lifespan:
                    print("Updating lifespan")
                    self.update_lifespan(ts)
                    self.recompute_lifespan = False
                xs, ys, ts, ps = xs[self.s_idx:-1], ys[self.s_idx:-1], ts[self.s_idx:-1], ps[self.s_idx:-1]
                ps = ps*100

            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps,
                    warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
            #print("iwe={}".format(np.sum(iwe)))

        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        loss = np.var(iwe-np.mean(iwe))
        #print(loss)
        return -loss

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*(g(x)-mu(g(x))*(g'(x)-mu(g'(x))) where g(x) is the IWE
        """
        if iwe is None or d_iwe is None:
            if self.adaptive_lifespan:
                if self.recompute_lifespan:
                    self.update_lifespan(ts)
                    self.recompute_lifespan = False
                xs, ys, ts, ps = xs[self.s_idx:-1], ys[self.s_idx:-1], ts[self.s_idx:-1], ps[self.s_idx:-1]
                ps = ps*100
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        zero_mean = 2.0*(iwe-np.mean(iwe))
        img_component = 2.0*(iwe-np.mean(iwe))
        for grad_dim in range(d_iwe.shape[0]):
            mean_jac = d_iwe[grad_dim]-np.mean(d_iwe[grad_dim])
            #gradient.append(np.mean(zero_mean*(d_iwe[grad_dim]-np.mean(d_iwe[grad_dim]))))
            #gradient.append((np.mean(zero_mean*(d_iwe[grad_dim]-np.mean(d_iwe[grad_dim])))))
            gradient.append(np.mean(img_component*d_iwe[grad_dim]))
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

    def __init__(self, adaptive_lifespan=False, minimum_events=10000):
        self.use_polarity = True
        self.name = "sos"
        self.has_derivative = True
        self.default_blur=1.0
        self.adaptive_lifespan = adaptive_lifespan
        self.pixel_crossings = 5
        self.minimum_events = minimum_events
        self.current_num_events = minimum_events
        self.div = 1

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by g(x)^2 where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
            iwe /= self.div
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        sos = np.mean(iwe*iwe)
        return -sos

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*g(x)*g'(x) where g(x) is IWE
        """
        if iwe is None or d_iwe is None:
            _, self.start = find_lifespan(ts, params, self.pixel_crossings)
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        img_component = (iwe*2.0)/(self.div*self.div)
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
