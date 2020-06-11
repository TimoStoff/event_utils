import argparse
import numpy as np
from representations.voxel_grid import events_to_neg_pos_voxel
from data_formats.read_events import read_h5_event_components
from visualization.draw_event_stream import plot_events
import matplotlib.pyplot as plt

def sample(cdf, ts):
    minval = cdf[0]
    maxval = cdf[-1]
    rnd = np.random.uniform(minval, maxval)
    idx = np.searchsorted(ts, rnd)
    return idx

def events_to_block(xs, ys, ts, ps):
    block_events = np.concatenate((
        xs[:,np.newaxis],
        ys[:,np.newaxis],
        ts[:,np.newaxis],
        ps[:,np.newaxis]), axis=1)
    return block_events

def merge_events(event_sets):
    xs,ys,ts,ps = [],[],[],[]
    for events in event_sets:
        xs.append(events[0])
        ys.append(events[1])
        ts.append(events[2])
        ps.append(events[3])
    merged = events_to_block(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
        np.concatenate(ps))
    return merged

def add_random_events(xs, ys, ts, ps, to_add, sort=True, return_merged=True):
    """
    Add fully random events
    :param xs: x component of events
    :param ys: y component of events
    :param ts: t component of events
    :param ps: p component of events
    :param to_add: how many events to add
    :param sort: sort the output events?
    :param return_merged: whether to return the random events separately or merged into
        the orginal input events
    """
    xs_new = np.random.randint(np.max(xs)+1, size=to_add)
    ys_new = np.random.randint(np.max(ys)+1, size=to_add)
    ts_new = np.random.uniform(np.min(ts), np.max(ts), size=to_add)
    ps_new = (np.random.randint(2, size=to_add))*2-1
    print(len(xs_new))
    if return_merged:
        new_events = merge_events([[xs_new, ys_new, ts_new, ps_new], [xs, ys, ts, ps]])
        if sort:
            new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
        return new_events[:,0], new_events[:,1], new_events[:,2], new_events[:,3],
    elif sort:
        new_events = events_to_block(xs_new, ys_new, ts_new, ps_new)
        new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
        return new_events[:,0], new_events[:,1], new_events[:,2], new_events[:,3],
    else:
        return xs_new, ys_new, ts_new, ps_new

def remove_events(xs, ys, ts, ps, to_remove):
    """
    Remove events randomly
    :param xs: x component of events
    :param ys: y component of events
    :param ts: t component of events
    :param ps: p component of events
    :param to_remove: how many events to remove
    """
    if to_remove > len(xs):
        return np.array([]), np.array([]), np.array([]), np.array([])
    to_select = len(xs)-to_remove
    idx = np.random.choice(np.arange(len(xs)), size=to_select, replace=False)
    idx.sort()
    return xs[idx], ys[idx], ts[idx], ps[idx]

def add_events(xs, ys, ts, ps, to_add, sort=True, return_merged=True, xy_std = 1.5, ts_std = 0.001):
    """
    Add events in the vicinity of existing events
    :param xs: x component of events
    :param ys: y component of events
    :param ts: t component of events
    :param ps: p component of events
    :param to_add: how many events to add
    :param sort: whether to sort the output events
    :param return_merged: whether to return the random events separately or merged into
        the orginal input events
    :xy_std: standard deviation of new xy coords
    :ts_std: standard deviation of new timestamp
    """
    iters = int(to_add/len(xs))+1
    xs_new, ys_new, ts_new, ps_new = [], [], [], []
    for i in range(iters):
        xs_new.append(xs+np.random.normal(scale=xy_std, size=xs.shape).astype(int))
        ys_new.append(ys+np.random.normal(scale=xy_std, size=ys.shape).astype(int))
        ts_new.append(ts+np.random.normal(scale=ts_std, size=ts.shape))
        ps_new.append(ps)
    xs_new = np.concatenate(xs_new, axis=0)
    ys_new = np.concatenate(ys_new, axis=0)
    ts_new = np.concatenate(ts_new, axis=0)
    ps_new = np.concatenate(ps_new, axis=0)
    idx = np.random.choice(np.arange(len(xs_new)), size=to_add, replace=False)
    if return_merged:
        new_events = merge_events([[xs_new, ys_new, ts_new, ps_new], [xs, ys, ts, ps]])
    else:
        new_events = events_to_block(xs_new, ys_new, ts_new, ps_new)
    if sort:
        new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
    return new_events[:,0], new_events[:,1], new_events[:,2], new_events[:,3],

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    parser.add_argument("--output_path", default="/tmp/extracted_data", help="Folder where to put augmented events")
    parser.add_argument("--to_add", type=float, default=1.0, help="How many more events, as a proportion \
            (eg, 1.5 will result in 150% more events, 0.2 will result in 20% of the events).")
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)
    num = 5000
    s = 10000
    num_to_add = 10000

    plot_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], elev=30, num_compress=1000, num_show=-1)

    nx, ny, nt, npo = add_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], num_to_add)
    plot_events(nx, ny, nt, npo, elev=30, num_compress=1000, num_show=-1)

    nx, ny, nt, npo = add_random_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], 5000)
    plot_events(nx, ny, nt, npo, elev=30, num_compress=1000, num_show=-1)

    nx, ny, nt, npo = remove_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], num//2)
    plot_events(nx, ny, nt, npo, elev=30, num_compress=1000, num_show=-1)
