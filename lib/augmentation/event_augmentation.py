import numpy as np
from lib.representations.voxel_grid import events_to_neg_pos_voxel
from lib.data_formats.read_events import read_h5_event_components
from lib.visualization.draw_event_stream import plot_events
from lib.util.event_util import clip_events_to_bounds
import matplotlib.pyplot as plt

def sample(cdf, ts):
    """
    Given a cumulative density function (CDF) and timestamps, draw
    a random sample from the CDF then find the index of the corresponding
    event. The idea is to allow fair sampling of an event streams timestamps
    @param cdf The CDF as np array
    @param ts The timestamps to sample from
    @returns The index of the sampled event
    """
    minval = cdf[0]
    maxval = cdf[-1]
    rnd = np.random.uniform(minval, maxval)
    idx = np.searchsorted(ts, rnd)
    return idx

def events_to_block(xs, ys, ts, ps):
    """
    Given events as lists of components, return a 4xN numpy array of the events
    where N is the number of events
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @returns The block of events
    """
    block_events = np.concatenate((
        xs[:,np.newaxis],
        ys[:,np.newaxis],
        ts[:,np.newaxis],
        ps[:,np.newaxis]), axis=1)
    return block_events

def merge_events(event_sets):
    """
    Merge multiple sets of events
    @param event_sets A list of event streams, where each event strea consists
        of four numpy arrays of xs, ys, ts and ps
    @returns One merged set of events as tuple: xs, ys, ts, ps
    """
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

def add_random_events(xs, ys, ts, ps, to_add, sensor_resolution=None,
        sort=True, return_merged=True):
    """
    Add new, random events drawn from a uniform distribution.
    Event coordinates are drawn from uniform dist over the sensor resolution and
    duration of the events.
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param to_add How many events to add
    @param sensor_resolution The resolution of the events. If left None, takes the range
        of the spatial coordinates of the imput events
    @param sort Sort the output events?
    @param return_merged Whether to return the random events separately or merged into
        the orginal input events
    @returns The random events as tuple: xs, ys, ts, ps
    """
    xs_new = np.random.randint(np.max(xs)+1, size=to_add)
    ys_new = np.random.randint(np.max(ys)+1, size=to_add)
    ts_new = np.random.uniform(np.min(ts), np.max(ts), size=to_add)
    ps_new = (np.random.randint(2, size=to_add))*2-1
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

def remove_events(xs, ys, ts, ps, to_remove, add_noise=0):
    """
    Remove events by random selection
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param to_remove How many events to remove
    @param add_noise How many noise events to add (0 by default)
    @returns Event stream with events removed as tuple: xs, ys, ts, ps
    """
    if to_remove > len(xs):
        return np.array([]), np.array([]), np.array([]), np.array([])
    to_select = len(xs)-to_remove
    idx = np.random.choice(np.arange(len(xs)), size=to_select, replace=False)
    if add_noise <= 0:
        idx.sort()
        return xs[idx], ys[idx], ts[idx], ps[idx]
    else:
        nsx, nsy, nst, nsp = add_random_events(xs, ys, ts, ps, add_noise, sort=False, return_merged=False)
        new_events = merge_events([[xs[idx], ys[idx], ts[idx], ps[idx]], [nsx, nsy, nst, nsp]])
        new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
        return new_events[:,0], new_events[:,1], new_events[:,2], new_events[:,3],

def add_correlated_events(xs, ys, ts, ps, to_add, sort=True, return_merged=True, xy_std = 1.5, ts_std = 0.001, add_noise=0):
    """
    Add events in the vicinity of existing events. Each original event has a Gaussian bubble
    placed around it from which the new events are sampled.
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param to_add How many events to add
    @param sort Whether to sort the output events
    @param return_merged Whether to return the random events separately or merged into
        the orginal input events
    @param xy_std Standard deviation of new xy coords
    @param ts_std standard deviation of new timestamp
    @param add_noise How many random noise events to add (default 0)
    @returns Events augemented with correlated events in tuple: xs, ys, ts, ps
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
    xs_new = np.clip(xs_new[idx], 0, np.max(xs))
    ys_new = np.clip(ys_new[idx], 0, np.max(ys))
    ts_new = ts_new[idx]
    ps_new = ps_new[idx]
    nsx, nsy, nst, nsp = add_random_events(xs, ys, ts, ps, add_noise, sort=False, return_merged=False)
    if return_merged:
        new_events = merge_events([[xs_new, ys_new, ts_new, ps_new], [nsx, nsy, nst, nsp]])
    else:
        new_events = events_to_block(xs_new, ys_new, ts_new, ps_new)
    if sort:
        new_events.view('i8,i8,i8,i8').sort(order=['f2'], axis=0)
    return new_events[:,0], new_events[:,1], new_events[:,2], new_events[:,3],

def flip_events_x(xs, ys, ts, ps, sensor_resolution=(180,240)):
    """
    Flip events along x axis
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @returns Flipped events
    """
    xs = sensor_resolution[1]-xs
    return xs, ys, ts, ps

def flip_events_y(xs, ys, ts, ps, sensor_resolution=(180,240)):
    """
    Flip events along y axis
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @returns Flipped events
    """
    ys = sensor_resolution[0]-ys
    return xs, ys, ts, ps

def crop_events(xs, ys, sensor_resolution, new_resolution):
    """
    Crop events to new resolution
    @param xs x component of events
    @param ys y component of events
    @param sensor_resolution Original resolution
    @param new_resolution New desired resolution
    @returns Events cropped to new resolution as tuple: xs, ys
    """
    clip = clip_events_to_bounds(xs, ys, None, None, new_resolution)
    return clip[0], clip[1]

def rotate_events(xs, ys, sensor_resolution=(180,240),
        theta_radians=None, center_of_rotation=None, clip_to_range=False):
    """
    Rotate events by a given angle around a given center of rotation.
    Note that the output events are floating point and may no longer
    be in the range of the image sensor. Thus, if 'standard' events are
    required, conversion to int and clipping to range may be necessary.
    @param xs x component of events
    @param ys y component of events
    @param sensor_resolution Size of event camera sensor
    @param theta_radians Angle of rotation in radians. If left empty, choose random
    @param center_of_rotation Center of the rotation. If left empty, choose random
    @param clip_to_range If True, remove events that lie outside of image plane after rotation
    @returns Rotated event coords and rotation parameters: xs, ys,
        theta_radians, center_of_rotation
    """
    theta_radians = np.random.uniform(0, 2*3.14159265359) if theta_radians is None else theta_radians
    corx = int(np.random.uniform(0, sensor_resolution[1])+1)
    cory = int(np.random.uniform(0, sensor_resolution[1])+1)
    center_of_rotation = (corx, cory) if center_of_rotation is None else center_of_rotation

    cxs = xs-center_of_rotation[0]
    cys = ys-center_of_rotation[1]
    new_xs = (cxs*np.cos(theta_radians)-cys*np.sin(theta_radians))+cxs
    new_ys = (cxs*np.sin(theta_radians)+cys*np.cos(theta_radians))+cys
    if clip_to_range:
        clip = clip_events_to_bounds(new_xs, new_ys, None, None, sensor_resolution)
        new_xs, new_ys = clip[0], clip[1]
    return new_xs, new_ys, theta_radians, center_of_rotation

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    parser.add_argument("--output_path", default="/tmp/extracted_data", help="Folder where to put augmented events")
    parser.add_argument("--to_add", type=float, default=1.0, help="How many more events, as a proportion \
            (eg, 1.5 will result in 150% more events, 0.2 will result in 20% of the events).")
    args = parser.parse_args()
    out_dir = args.output_path

    xs, ys, ts, ps = read_h5_event_components(args.path)
    ys = 180-ys
    num = 50000
    s = 0#10000
    num_to_add = num*2
    num_comp=5000

    pth = os.path.join(out_dir, "img0")
    plot_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)

    pth = os.path.join(out_dir, "img1")
    nx, ny, nt, npo = add_correlated_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], num_to_add)
    plot_events(nx, ny, nt, npo, elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)

    pth = os.path.join(out_dir, "img3")
    nx, ny, nt, npo = add_random_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], num_to_add, sensor_resolution=(180,240))
    plot_events(nx, ny, nt, npo, elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)

    pth = os.path.join(out_dir, "img4")
    nx, ny, nt, npo = remove_events(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num], num//2)
    plot_events(nx, ny, nt, npo, elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)

    pth = os.path.join(out_dir, "img5")
    nx, ny, rot, cor = rotate_events(xs[s:s+num], ys[s:s+num], theta_radians=1.4, center_of_rotation=(90, 120), clip_to_range=True)
    plot_events(nx, ny, ts, ps, elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)

    pth = os.path.join(out_dir, "img6")
    nx, ny, rot, cor = flip_events_x(xs[s:s+num], ys[s:s+num], ts[s:s+num], ps[s:s+num])
    plot_events(nx, ny, ts, ps, elev=30, num_compress=num_comp, num_show=-1, save_path=pth, show_axes=True, compress_front=True)
