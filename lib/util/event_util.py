import numpy as np
import h5py

def infer_resolution(xs, ys):
    sr = [np.max(ys) + 1, np.max(xs) + 1]
    return sr

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def clip_events_to_bounds(xs, ys, ts, ps, bounds, set_zero=False):
    """
    Clip events to the given bounds.
    :param: xs x coords of events
    :param: ys y coords of events
    :param: ts t coords of events (may be None)
    :param: ps p coords of events (may be None)
    :param: bounds the bounds of the events. Must be list of
        length 2 (in which case the lower bound is assumed to be 0,0)
        or length 4, in format [min_y, max_y, min_x, max_x]
    :param: set_zero if True, simply multiplies the out of bounds events with 0 mask.
        Otherwise, removes the events.
    """
    if len(bounds) == 2:
        bounds = [0, bounds[0], 0, bounds[1]]
    elif len(bounds) != 4:
        raise Exception("Bounds must be of length 2 or 4 (not {})".format(len(bounds)))
    miny, maxy, minx, maxx = bounds
    if set_zero:
        mask = events_bounds_mask(xs, ys, minx, maxx, miny, maxy)
        ts_mask = None if ts is None else ts*mask
        ps_mask = None if ps is None else ps*mask
        return xs*mask, ys*mask, ts_mask, ps_mask
    else:
        x_clip_idc = np.argwhere((xs >= minx) & (xs < maxx))[:, 0]
        y_subset = ys[x_clip_idc]
        y_clip_idc = np.argwhere((y_subset >= miny) & (y_subset < maxy))[:, 0]

        xs_clip = xs[x_clip_idc][y_clip_idc]
        ys_clip = ys[x_clip_idc][y_clip_idc]
        ts_clip = None if ts is None else ts[x_clip_idc][y_clip_idc]
        ps_clip = None if ps is None else ps[x_clip_idc][y_clip_idc]
        return xs_clip, ys_clip, ts_clip, ps_clip

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def binary_search_h5_timestamp(hdf_path, l, r, x, side='left'):
    f = h5py.File(hdf_path, 'r')
    return binary_search_h5_dset(f['events/ts'], x, l=l, r=r, side=side)

def binary_search_torch_tensor(t, l, r, x, side='left'):
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r
