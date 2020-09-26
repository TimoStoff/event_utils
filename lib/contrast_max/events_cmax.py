import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
import torch
import copy
from ..util.event_util import infer_resolution
from ..util.util import plot_image, save_image, plot_image_grid
from ..visualization.draw_event_stream import plot_events
from .objectives import *
from .warps import *

def grid_cmax(xs, ys, ts, ps, roi_size=(20,20), step=None, warp=linvel_warp(),
        obj=variance_objective(adaptive_lifespan=True, minimum_events=105)):
    step = roi_size if step is None else step
    resolution = infer_resolution(xs, ys)
    warpfunc = linvel_warp()

    results_params = []
    results_rois = []
    results_f_evals = []
    samplenum = 0
    min_events = 10
    print("Running grid search...")
    for xc in range(0, resolution[1], step[1]):
        x_roi_idc = np.argwhere((xs>=xc) & (xs<xc+step[1]))[:, 0]
        y_subset = ys[x_roi_idc]
        for yc in range(0, resolution[0], step[0]):
            y_roi_idc = np.argwhere((y_subset>=yc) & (y_subset<yc+step[0]))[:, 0]

            roi_xs = xs[x_roi_idc][y_roi_idc]
            roi_ys = ys[x_roi_idc][y_roi_idc]
            roi_ts = ts[x_roi_idc][y_roi_idc]
            roi_ps = ps[x_roi_idc][y_roi_idc]

            if len(roi_xs) > min_events:
                #params = optimize(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False)
                obj = variance_objective(adaptive_lifespan=True, minimum_events=105)
                print("OPTIMIZE using {} events".format(len(roi_xs)))
                params = optimize_contrast(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False, blur_sigma=2.0, img_size=resolution, grid_search_init=True)
                print("--- first cycle ---")
                params = optimize_contrast(roi_xs, roi_ys, roi_ts, roi_ps, warp, obj, numeric_grads=False, blur_sigma=1.0, img_size=resolution, x0=params)
                print("Final optimization param: {}".format(params))
                print("===============")
                iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warp, resolution,
                       use_polarity=True, compute_gradient=False, return_events=False)
                f_eval = obj.evaluate_function(iwe=iwe)

                samplenum += 1
                results_params.append(params)
                results_rois.append([yc, xc, step[0], step[1]])
                results_f_evals.append(obj.evaluate_function(iwe=iwe))

                #return [params], 0
                ##xsc, ysc, tsc, psc = cut_events_to_lifespan(xs, ys, ts, ps, params, 10, minimum_events=10)
                #xsc, ysc, tsc, psc = xs, ys, ts, ps
                #iwe, d_iwe, (xw, ywe) = get_iwe(params, xsc, ysc, tsc, psc, warp, resolution,
                #       use_polarity=True, compute_gradient=True, return_events=True)
                #plot_image(iwe, bbox=[xc,yc,step[1],step[0]], savename="/tmp/iwe_{}.jpg".format(0))
                #plot_image_grid([d_iwe[0], d_iwe[1]], norm=False, savename="/tmp/diwe_{}.jpg".format(0))

                # Old segmentation
                #for i in range(1, 100, 10):
                #    img1, d_iwe, (wx,wy), ew1 = get_iwe(params, xsc, ysc, tsc, psc, warp, resolution,
                #           use_polarity=True, return_per_event_contrast=True, return_events=True)
                #    img1 = events_to_image(wx, wy, psc, interpolation='bilinear', padding=True)
                #    plot_image(img1, savename="/tmp/img1_{}.jpg".format(i))
                #    ew1 = image_to_event_weights(wx, wy, img1)
                #    params2 = params + np.array([i, i])
                #    print(params2)
                #    img2, d_iwe, ew2 = get_iwe(params2, xsc, ysc, tsc, psc, warp, resolution,
                #           use_polarity=True, return_per_event_contrast=True)
                #    img3 = img1-img2
                #    dew = ew1-ew2

                #    dew_iwe, d_iwe = get_iwe(params2, xsc, ysc, tsc, dew, warp, resolution,
                #           use_polarity=True)
                #    re_iwe = events_to_image(wx, wy, ew1, interpolation='bilinear')
                #    plot_image(re_iwe, savename="/tmp/re_iwe_{}.jpg".format(i))

                #    plot_image_grid([img1, img2, img3], norm=False, savename="/tmp/img_comp{}.jpg".format(i))
                #    save_image(img3, fname="/tmp/img_{}.jpg".format(i))
                #    save_image(dew_iwe, fname="/tmp/dew_{}.jpg".format(i))



                ##seg_mask = gaussian_filter(segmentation_mask_from_d_iwe(d_iwe), 1.0)
                #seg_mask = segmentation_mask_from_d_iwe(d_iwe).astype(np.float)
                #plot_image(seg_mask)
                #seg_mask = gaussian_filter(seg_mask, sigma=1)

                #event_ind = get_events_from_mask(seg_mask, xw, yw)
                #print(event_ind.shape)
                #segx = xsc[event_ind[0:10000]]
                #segy = ysc[event_ind[0:10000]]
                #segt = tsc[event_ind[0:10000]]
                #segp = psc[event_ind[0:10000]]
                #img = [np.ones((180,240), dtype=np.float)]
                #img_ts = [segt[0]]
                #plot_events(240-segx, 180-segy, segt, segp, show_plot=True, num_show=-1, img_size=(180,240), num_compress=1000)
                #print(len(segx))

    return results_params, results_rois, results_f_evals

def segmentation_mask_from_d_iwe(d_iwe, th=None):
    th1 = np.percentile(np.abs(d_iwe), 90)
    validx = d_iwe[0].flatten()[np.argwhere(np.abs(d_iwe[0].flatten()) > th1).squeeze()]
    validy = d_iwe[1].flatten()[np.argwhere(np.abs(d_iwe[1].flatten()) > th1).squeeze()]
    x_c = np.percentile(validx, 95)
    y_c = np.percentile(validy, 95)

    thx = x_c if th is None else th
    thy = y_c if th is None else th

    imgxp = np.where(d_iwe[0] > thx, 1, 0)
    imgyp = np.where(d_iwe[1] > thy, 1, 0)
    imgxn = np.where(d_iwe[0] < -thx, 1, 0)
    imgyn = np.where(d_iwe[1] < -thy, 1, 0)
    imgx = imgxp + imgxn
    imgy = imgyp + imgyn
    img = np.clip(np.add(imgx, imgy), 0, 1)
    return img

def draw_objective_function(xs, ys, ts, ps, objective, warpfunc, x_range=(-200, 200), y_range=(-200, 200),
        gt=(0,0), show_gt=True, resolution=20, img_size=(180, 240), show_axes=True, norm_min=None, norm_max=None,
        show=True):
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
    norm_min = np.min(img) if norm_min is None else norm_min
    norm_max = np.max(img) if norm_max is None else norm_max
    img = (img-norm_min)/((norm_max-norm_min)+1e-6)
    #img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img, interpolation='bilinear', cmap='viridis')
    if not show_axes:
        plt.xticks([])
        plt.yticks([])
    else:
        xt = plt.xticks()[0][1:-1]
        xticklabs = np.linspace(x_range[0], x_range[1], len(xt))
        xticklabs = ["{}".format(int(x)) for x in xticklabs]

        yt = plt.yticks()[0][1:-1]
        yticklabs = np.linspace(y_range[0], y_range[1], len(yt))
        yticklabs = ["{}".format(int(y)) for y in yticklabs]

        plt.xticks(ticks=xt, labels=xticklabs)
        plt.yticks(ticks=yt, labels=yticklabs)

        plt.xlabel("$v_x$")
        plt.ylabel("$v_y$")

    if show_gt:
        xloc = ((gt[0]-x_range[0])/(width))*imshape[1]
        yloc = ((gt[1]-y_range[0])/(height))*imshape[0]
        plt.axhline(y=yloc, color='r', linestyle='--')
        plt.axvline(x=xloc, color='r', linestyle='--')
    if show:
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
        log_scale=True, num_samples_per_param=5, depth=0, th0=1, max_iters=20):
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
    optimal = grid_search_initial(xs, ys, ts, ps, warp_function, copy.deepcopy(objective_function),
            img_size, param_ranges=param_ranges, log_scale=log_scale,
            num_samples_per_param=num_samples_per_param)

    params = optimal["min_params"]
    new_param_ranges = []
    max_range = 0
    for sa, param in zip(optimal["search_axes"], params):
        new_range = find_new_range(sa, param)
        new_param_ranges.append(new_range)
        max_range = np.abs(new_range[1]-new_range[0]) if np.abs(new_range[1]-new_range[0]) > max_range else max_range
    #print("--- Depth={}, range={} ---".format(depth, max_range))
    if max_range >= th0 and depth < max_iters:
        return recursive_search(xs,ys,ts,ps,warp_function,objective_function,img_size,
                param_ranges=new_param_ranges, log_scale=log_scale,
                num_samples_per_param=num_samples_per_param, depth=depth+1)
    else:
        #print("SOFAS search: {}".format(optimal["min_params"]))
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
            param_ranges.append([-150, 150])

    #draw_objective_function(xs, ys, ts, ps, variance_objective(), linvel_warp(), x_range=param_ranges[0], y_range=param_ranges[1], show_gt=False, resolution=2)
    #old_xlim = plt.xlim()
    #old_ylim = plt.ylim()

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

    #xpts, ypts = [], []
    for params in zip(*coords):
        f_eval = objective_function.evaluate_function(params=params, xs=xs, ys=ys, ts=ts, ps=ps,
                warpfunc=warp_function, img_size=img_size, blur_sigma=1.0)
        #xpts.append(params[0])
        #ypts.append(params[1])
        #print("{}: {}".format(params, f_eval))
        output["params"].append(params)
        output["eval"].append(f_eval)
        if f_eval < best_eval:
            best_eval = f_eval
            best_params = params

    #xpts = (xpts-min(xpts))/(max(xpts)-min(xpts))*(old_xlim[1]-old_xlim[0])+old_xlim[0]
    #ypts = (ypts-min(ypts))/(max(ypts)-min(ypts))*(old_ylim[0]-old_ylim[1])+old_ylim[1]
    #plt.xlim(old_xlim)
    #plt.ylim(old_ylim)
    #plt.scatter(xpts, ypts, c='r', marker='x')
    #plt.show()

    output["min_params"] = best_params
    output["min_func_eval"] = best_eval
    #print("min @ {}: {}".format(best_params, best_eval))
    return output

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240), grid_search_init=False, minimum_events=200):
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
        init_obj = copy.deepcopy(objective)
        init_obj.adaptive_lifespan = False
        minv = recursive_search(xs, ys, ts, ps, warp_function, init_obj, img_size, log_scale=False)
        #xs, ys, ts, ps = cut_events_to_lifespan(xs, ys, ts, ps, minv["min_params"], 5, minimum_events=minimum_events)
        #minv = grid_search_initial(xs, ys, ts, ps, warp_function, objective, img_size, log_scale=False)
        x0 = minv["min_params"]
    elif x0 is None:
        x0 = np.array([0,0])
    print("Using {} events, x0 = {}".format(len(xs), x0))
    objective.iter_update(x0)
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False, callback=objective.iter_update)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=True, callback=objective.iter_update)
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
