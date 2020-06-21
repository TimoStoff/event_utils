import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
import torch
from ..util.event_util import infer_resolution
from ..util.util import plot_image, save_image
from .objectives import *
from .warps import *

def grid_cmax(xs, ys, ts, ps, roi_size=(20,20), step=None, warp=linvel_warp(), obj=variance_objective()):
    step = roi_size if step is None else step
    resolution = infer_resolution(xs, ys)
    warpfunc = linvel_warp()

    results_params = []
    results_rois = []
    samplenum = 0
    for xc in range(0, resolution[1], step[1]):
        x_roi_idc = np.argwhere((xs>=xc) & (xs<xc+step[1]))[:, 0]
        y_subset = ys[x_roi_idc]
        for yc in range(0, resolution[0], step[0]):
            bbox = [(xc, yc), (xc+step[1], yc+step[0])]
            y_roi_idc = np.argwhere((y_subset>=yc) & (y_subset<yc+step[0]))[:, 0]

            roi_xs = xs[x_roi_idc][y_roi_idc]
            roi_ys = ys[x_roi_idc][y_roi_idc]
            roi_ts = ts[x_roi_idc][y_roi_idc]
            roi_ps = ps[x_roi_idc][y_roi_idc]

            if len(roi_xs) > 0:
                #params = optimize(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False)
                params = optimize_contrast(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False, blur_sigma=2.0, img_size=resolution, grid_search_init=True)
                params = optimize_contrast(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False, blur_sigma=1.0, img_size=resolution, x0=params)
                iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warp, resolution,
                       use_polarity=True, compute_gradient=False)
                save_image(iwe, fname="/tmp/img_{:09d}.png".format(samplenum), bbox=[[xc,yc],[xc+step[1],yc+step[0]]])
                samplenum += 1
                results_params.append(params)
                results_rois.append([yc, xc, yc+step[0], xc+step[1]])
    return results_params, results_rois

def draw_objective_function(xs, ys, ts, ps, objective, warpfunc, x_range=(-200, 200), y_range=(-200, 200),
        gt=(0,0), show_gt=True, resolution=20, img_size=(180, 240)):
    """
    Draw the objective function given by sampling over a range. Depending on the value of resolution, this
    can involve many samples and take some time.
    Parameters:
        xs,ys,ts,ps (numpy array) The event components
        objective (object) The objective function
        warpfunc (object) The warp function
        x_range, y_range (tuple) the range over which to plot the parameters
        gt (tuple) The ground truth
        show_gt (bool) Whether to draw the ground truth in
        resolution (float) The resolution of the sampling
        img_size (tuple) The image sensor size
    """
    width = x_range[1]-x_range[0]
    height = y_range[1]-y_range[0]
    print("Drawing objective function. Taking {} samples".format((width*height)/resolution))
    imshape = (int(height/resolution+0.5), int(width/resolution+0.5))
    img = np.zeros(imshape)
    for x in range(img.shape[1]):
       for y in range(img.shape[0]):
           params = np.array([x*resolution+x_range[0], y*resolution+y_range[0]])
           img[y,x] = -objective.evaluate_function(params, xs, ys, ts, ps, warpfunc, img_size, blur_sigma=0)
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img, interpolation='bilinear', cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    if show_gt:
        xloc = ((gt[0]-x_range[0])/(width))*imshape[1]
        yloc = ((gt[1]-y_range[0])/(height))*imshape[0]
        plt.axhline(y=yloc, color='r', linestyle='--')
        plt.axvline(x=xloc, color='r', linestyle='--')
    plt.show()

def find_new_range(search_axes, param):
    """
    Given a range of search parameters and a parameter, find
    the new range that encompasses all unsearched domain around
    the parameter
    """
    magnitude = np.abs(param)
    nearest_idx = np.searchsorted(search_axes, param)
    if nearest_idx >= len(search_axes)-1:
        d1 = np.abs(search_axes[-1]-search_axes[-2])
        d2 = d1
    elif nearest_idx == 0:
        d1 = np.abs(search_axes[0]-search_axes[-1])
        d2 = np.abs(search_axes[0]-search_axes[1])
    else:
        d1 = np.abs(search_axes[nearest_idx]-search_axes[nearest_idx-1])
        d2 = np.abs(search_axes[nearest_idx]-search_axes[nearest_idx+1])
    param_range = [param-d1, param+d2]
    return param_range

def recursive_search(xs, ys, ts, ps, warp_function, objective_function, img_size, param_ranges=None,
        log_scale=True, num_samples_per_param=5, depth=0, th0=10, max_iters=10):
    """
    Recursive grid-search optimization as per SOFAS. Searches a grid over a range
    and then searches a sub-grid, etc, until convergence.

    :param: xs x components of events
    :param: ys y components of events
    :param: ts t components of events
    :param: ps p components of events
    :param: warp_function the warp function to use
    :param: objective_function the objective function to use
    :param: img_size the size of the event camera sensor
    :param: param_ranges: a list of lists, where each list contains the search range for
        the given warp function parameter. If None, the default is to search from -100 to 100 for
        each parameter.
    :param: log_scale if true, the sample points are drawn from a log scale. This means that
        the parameter space is searched more frequently near the origin and less frequently at
        the fringes.
    :param: num_samples_per_param how many samples to take per parameter. The number of evaluations
        this method needs to perform is equal to num_samples_per_param^warp_function.dims. Thus,
        for high dimensional warp functions, it is advised to keep this value low. Must be greater
        than 5 and odd.
    :param: depth keeps track of the recursion depth
    :param: th0 when the subgrid search radius is smaller than th0, convergence is reached.
    :param: max_iters maximum number of iterations
    """
    assert num_samples_per_param%2==1 and num_samples_per_param>=5
    optimal = grid_search_initial(xs, ys, ts, ps, warp_function, objective_function,
            img_size, param_ranges=param_ranges, log_scale=log_scale,
            num_samples_per_param=num_samples_per_param)

    params = optimal["min_params"]
    new_param_ranges = []
    max_range = 0
    for sa, param in zip(optimal["search_axes"], params):
        new_range = find_new_range(sa, param)
        new_param_ranges.append(new_range)
        max_range = np.abs(new_range[1]-new_range[0]) if np.abs(new_range[1]-new_range[0]) > max_range else max_range
    print("--- Depth={}, range={} ---".format(depth, max_range))
    if max_range >= th0 and depth < max_iters:
        return recursive_search(xs,ys,ts,ps,warp_function,objective_function,img_size,
                param_ranges=new_param_ranges, log_scale=log_scale,
                num_samples_per_param=num_samples_per_param, depth=depth+1)
    else:
        print("SOFAS search: {}".format(optimal["min_params"]))
        return optimal



def grid_search_initial(xs, ys, ts, ps, warp_function, objective_function, img_size, param_ranges=None,
        log_scale=True, num_samples_per_param=5):
    """
    Perform a grid search for a good starting candidate.
    :param: xs x components of events
    :param: ys y components of events
    :param: ts t components of events
    :param: ps p components of events
    :param: warp_function the warp function to use
    :param: objective_function the objective function to use
    :param: img_size the size of the event camera sensor
    :param: param_ranges: a list of lists, where each list contains the search range for
        the given warp function parameter. If None, the default is to search from -100 to 100 for
        each parameter.
    :param: log_scale if true, the sample points are drawn from a log scale. This means that
        the parameter space is searched more frequently near the origin and less frequently at
        the fringes.
    :param: num_samples_per_param how many samples to take per parameter. The number of evaluations
        this method needs to perform is equal to num_samples_per_param^warp_function.dims. Thus,
        for high dimensional warp functions, it is advised to keep this value low.
    """
    assert num_samples_per_param%2 == 1

    if log_scale:
        #Function is sampled from 10^x from 0 to 2
        scale = np.logspace(0, 2.0, int(num_samples_per_param/2.0)+1)[1:]
        scale /= scale[-1]
    else:
        scale = np.linspace(0, 1.0, int(num_samples_per_param/2.0)+1)[1:]

    if param_ranges is None:
        param_ranges = []
        for i in range(warp_function.dims):
            param_ranges.append([-100, 100])

    axes = []
    for param_range in param_ranges:
        rng = param_range[1]-param_range[0]
        mid = param_range[0] + rng/2.0
        rescale_pos = np.array(mid+scale*(rng/2.0))
        rescale_neg = np.array(mid-scale*(rng/2.0))[::-1]
        rescale = np.concatenate((rescale_neg, np.array([mid]), rescale_pos))
        axes.append(rescale)
    grids = np.meshgrid(*axes)
    coords = np.vstack(map(np.ravel, grids))

    output = {"params":[], "eval": [], "search_axes": axes}
    best_eval = 0
    best_params = None
    for params in zip(*coords):
        f_eval = objective_function.evaluate_function(params=params, xs=xs, ys=ys, ts=ts, ps=ps,
                warpfunc=warp_function, img_size=img_size, blur_sigma=1.0)
        #print("{}: {}".format(params, f_eval))
        output["params"].append(params)
        output["eval"].append(f_eval)
        if f_eval < best_eval:
            best_eval = f_eval
            best_params = params
    output["min_params"] = best_params
    output["min_func_eval"] = best_eval
    #print("min @ {}: {}".format(best_params, best_eval))
    return output

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240), grid_search_init=False):
    """
    Optimize contrast for a set of events
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp_function (function) The function with which to warp the events
    objective (objective class object) The objective to optimize
    optimizer (function) The optimizer to use
    x0 (np array) The initial guess for optimization
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor
    blur_sigma (float) Size of the blurring kernel. Blurring the images of warped events can
        have a large impact on the convergence of the optimization.

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    if grid_search_init and x0 is None:
        print("-----------")
        minv = recursive_search(xs, ys, ts, ps, warp_function, objective, img_size, log_scale=False)
        #minv = grid_search_initial(xs, ys, ts, ps, warp_function, objective, img_size, log_scale=False)
        x0 = minv["min_params"]
        print("x0 at {}".format(x0))
    elif x0 is None:
        x0 = np.array([0,0])
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events. Uses optimize_contrast() for the optimiziation, but allows
    blurring schedules for successive optimization iterations.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=1.0, img_size=img_size)
    return argmax_an

def optimize_r2(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events, finishing with SoE loss.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    soe_obj = soe_objective()
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
    return argmax_an

if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        gt Ground truth optic flow for event slice
        img_size The size of the event camera sensor
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts-ts[0]
    gt_params = tuple(args.gt)
    img_size=tuple(args.img_size)

    start_idx = 20000
    end_idx=start_idx+15000
    blur = None

    draw_objective_function(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], variance_objective(), linvel_warp())

    objectives = [r1_objective(), zhu_timestamp_objective(), variance_objective(), sos_objective(), soe_objective(), moa_objective(),
            isoa_objective(), sosa_objective(), rms_objective()]
    warp = linvel_warp()
    for obj in objectives:
        argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], warp, obj, numeric_grads=True)
        loss = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                ps[start_idx:end_idx], warp, img_size=img_size)
        gtloss = obj.evaluate_function(gt_params, xs[start_idx:end_idx], ys[start_idx:end_idx],
                ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
        print("{}:({})={}, gt={}".format(obj.name, argmax, loss, gtloss))
        if obj.has_derivative:
            argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                    ps[start_idx:end_idx], warp, obj, numeric_grads=False)
            loss_an = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx],
                    ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
            print("   analytical:{}={}".format(argmax, loss_an))
