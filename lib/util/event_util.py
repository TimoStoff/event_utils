import numpy as np
import h5py
from ..representations.image import events_to_image

def infer_resolution(xs, ys):
    """
    Given events, guess the resolution by looking at the max and min values
    @param xs Event x coords
    @param ys Event y coords
    @returns Inferred resolution
    """
    sr = [np.max(ys) + 1, np.max(xs) + 1]
    return sr

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    @param xs Event x coords
    @param ys Event y coords
    @param x_min Lower bound of x axis
    @param x_max Upper bound of x axis
    @param y_min Lower bound of y axis
    @param y_max Upper bound of y axis
    @returns mask
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def cut_events_to_lifespan(xs, ys, ts, ps, params,
        pixel_crossings, minimum_events=100, side='back'):
    """
    Given motion model parameters, compute the speed and thus
    the lifespan, given a desired number of pixel crossings
    @param xs Event x coords
    @param ys Event y coords
    @param ts Event timestamps
    @param ps Event polarities
    @param params Motion model parameters
    @param pixel_crossings Number of pixel crossings
    @param minimum_events The minimum number of events to cut down to
    @param side Cut events from 'back' or 'front'
    @returns Cut events
    """
    magnitude = np.linalg.norm(params)
    dt = pixel_crossings/magnitude
    if side == 'back':
        s_idx = np.searchsorted(ts, ts[-1]-dt)
        num_events = len(xs)-s_idx
        s_idx = len(xs)-minimum_events if num_events < minimum_events else s_idx
        return xs[s_idx:-1], ys[s_idx:-1], ts[s_idx:-1], ps[s_idx:-1]
    elif side == 'front':
        s_idx = np.searchsorted(ts, dt+ts[0])
        num_events = s_idx
        s_idx = minimum_events if num_events < minimum_events else s_idx
        return xs[0:s_idx], ys[0:s_idx], ts[0:s_idx], ps[0:s_idx]
    else:
        raise Exception("Invalid side given: {}. To cut events, must provide an \
                appropriate side to cut from, either 'front' or 'back'".format(side))

def clip_events_to_bounds(xs, ys, ts, ps, bounds, set_zero=False):
    """
    Clip events to the given bounds.
    @param xs x coords of events
    @param ys y coords of events
    @param ts Timestamps of events (may be None)
    @param ps Polarities of events (may be None)
    @param bounds the bounds of the events. Must be list of
       length 2 (in which case the lower bound is assumed to be 0,0)
       or length 4, in format [min_y, max_y, min_x, max_x]
    @param: set_zero if True, simply multiplies the out of bounds events with 0 mask.
        Otherwise, removes the events.
    @returns Clipped events
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

def get_events_from_mask(mask, xs, ys):
    """
    Given an image mask, return the indices of all events at each location in the mask
    @params mask The image mask
    @param xs x components of events as list
    @param ys y components of events as list
    @returns Indices of events that lie on the mask
    """
    xs = xs.astype(int)
    ys = ys.astype(int)
    idx = np.stack((ys, xs))
    event_vals = mask[tuple(idx)]
    event_indices = np.argwhere(event_vals >= 0.01).squeeze()
    return event_indices

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The HDF5 dataset
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
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
    """
    Binary search implemented for pytorch tensors (no native implementation exists)
    @param t The tensor
    @param x The value being searched for
    @param l Starting lower bound (0 if None is chosen)
    @param r Starting upper bound (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
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

def remove_hot_pixels(xs, ys, ts, ps, sensor_size=(180, 240), num_hot=50):
    """
    Given a set of events, removes the 'hot' pixel events.
    Accumulates all of the events into an event image and removes
    the 'num_hot' highest value pixels.
    @param xs Event x coords
    @param ys Event y coords
    @param ts Event timestamps
    @param ps Event polarities
    @param sensor_size The size of the event camera sensor
    @param num_hot The number of hot pixels to remove
    """
    img = events_to_image(xs, ys, ps, sensor_size=sensor_size)
    hot = np.array([])
    for i in range(num_hot):
        maxc = np.unravel_index(np.argmax(img), sensor_size)
        #print("{} = {}".format(maxc, img[maxc]))
        img[maxc] = 0
        h = np.where((xs == maxc[1]) & (ys == maxc[0]))
        hot = np.concatenate((hot, h[0]))
    xs, ys, ts, ps = np.delete(xs, hot), np.delete(ys, hot), np.delete(ts, hot), np.delete(ps, hot)
    return xs, ys, ts, ps
