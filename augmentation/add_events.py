import argparse
import numpy as np
from representations.voxel_grid import events_to_neg_pos_voxel
from data_formats.read_events import read_h5_event_components
import matplotlib.pyplot as plt

def sample(cdf, ts):
    minval = cdf[0]
    maxval = cdf[-1]
    rnd = np.random.uniform(minval, maxval)
    idx = np.searchsorted(ts, rnd)
    return idx

def add_events_d(xs, ys, ts, ps, to_add, sensor_size=(180,240)):
    xy_std = 1.0
    ts_std = 0.0001
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
    idx = np.random.choice(np.arange(len(xs)), size=to_add, replace=False)
    new_events = np.concatenate((
        np.concatenate((xs_new[idx], xs))[:,np.newaxis],
        np.concatenate((ys_new[idx], ys))[:,np.newaxis],
        np.concatenate((ts_new[idx], ts))[:,np.newaxis],
        np.concatenate((ps_new[idx], ps))[:,np.newaxis]), axis=1)
    new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
    return new_events

def add_events(xs, ys, ts, ps, proportion, bins=20, sensor_size=(180,240)):
    """
    Add events:
        1: Create voxel grid
        2: Norm voxel grid. Voxels are now probabilities
        3: For each event, sample its probability p. p*proportion is now the
            probability of spawning a new event (in a Gaussian around
            the event).
    """
    vg_pos, vg_neg = events_to_neg_pos_voxel(xs, ys, ts, ps, bins, sensor_size)
    for i in range(bins):
        print("{}: {}".format(i, np.sum(vg_pos[i,:,:])))

    vg_sum = np.sum(vg_pos) + np.sum(vg_neg)
    vg_pos /= vg_sum
    vg_neg /= vg_sum

    dt = ts[-1]-ts[0]
    t_bin = (ts-ts[0])/dt*(bins-1)
    tb_idx = t_bin.astype(int)
    tb_idx[-1] -= 1

    #print(tb_idx)
    #print(t_bin)
    #print(vg_pos[tb_idx, ys, xs])
    #print(vg_pos[tb_idx+1, ys, xs])
    p0=1.0-t_bin%1
    p1=t_bin%1
    #for i, (x,y,pr0,pr1) in enumerate(zip(vg_pos[tb_idx, ys, xs], vg_pos[tb_idx+1, ys, xs], p0, p1)):
    #    print("{}: {},{}={}, p0={}, p1={}".format(i, x, y, x+y, pr0, pr1))
    l_pos = (vg_pos[tb_idx, ys, xs]*(1.0-t_bin%1) + vg_pos[tb_idx+1, ys, xs]*(t_bin%1))
    l_neg = (vg_neg[tb_idx, ys, xs]*(1.0-t_bin%1) + vg_neg[tb_idx+1, ys, xs]*(t_bin%1))
    pos_mask, neg_mask = np.where(ps, 1, 0), np.where(ps, 0, 1)
    likelihoods = (l_pos*pos_mask)+(l_neg*neg_mask)
    l_cum = np.cumsum(likelihoods)
    #print(likelihoods)
    #print(l_cum)

    xax = np.arange(len(l_cum))
    fig = plt.figure()
    plt.scatter(xax, l_cum)
    plt.show()

    new_events = {'xs':[], 'ys':[], 'ts':[], 'ps':[]}

    for x,y,t,p,l in zip(xs, ys, ts, ps, likelihoods):
        pass

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    parser.add_argument("--output_path", default="/tmp/extracted_data", help="Folder where to put augmented events")
    parser.add_argument("--to_add", type=float, default=1.0, help="Roughly how many more events, as a proportion \
            (eg, 1.5 will results in approximately 150% more events.")
    parser.add_argument('--exact', action='store_true', help="If true, will create exactly --to_add events")
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)
    num = 5000
    s = 10000
    add_events_d(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], 5)
