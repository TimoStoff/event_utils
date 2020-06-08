import numpy as np

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
