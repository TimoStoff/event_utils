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

def clip_events_to_bounds(xs, ys, ps, bounds):
    """
    Clip events to the given bounds
    """
    mask = events_bounds_mask(xs, ys, 0, bounds[1], 0, bounds[0])
    return xs*mask, ys*mask, ps*mask

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
