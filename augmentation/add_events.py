import argparse
import numpy as np
from representations.voxel_grid import events_to_neg_pos_voxel
from data_formats.read_events import read_h5_event_components

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
    vg_sum = np.sum(vg_pos) + np.sum(vg_neg)
    vg_pos /= vg_sum
    vg_neg /= vg_sum

    dt = ts[-1]-ts[0]
    t_bin = (ts-ts[0])/dt*(bins-1)
    tb_idx = t_bin.astype(int)

    print(vg_pos.shape)
    a = vg_pos[tb_idx, ys, xs]
    b = (1.0-t_bin%1)
    print(a.shape)
    print(b)
    print(tb_idx)
    print(t_bin)
    l_pos = (vg_pos[tb_idx, ys, xs]*(1.0-t_bin%1) + vg_pos[tb_idx+1, ys, xs]*(t_bin%1))
    l_neg = (vg_neg[tb_idx, ys, xs]*(1.0-t_bin%1) + vg_neg[tb_idx+1, ys, xs]*(t_bin%1))
    pos_mask, neg_mask = np.where(ps, 1, 0), np.where(ps, 0, 1)
    likelihoods = (l_pos*pos_mask)+(l_neg*neg_mask)
    print(likelihoods)

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
    num = 10000
    s = 10000
    add_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], 1.5, bins=10)
