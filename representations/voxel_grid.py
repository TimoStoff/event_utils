import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
from util.event_util import clip_events_to_bounds, events_bounds_mask
from .image import events_to_image, events_to_image_torch

def get_voxel_grid_as_image(voxelgrid):
    images = []
    splitter = np.ones((voxelgrid.shape[1], 2))*np.max(voxelgrid)
    for image in voxelgrid:
        images.append(image)
        images.append(splitter)
    images.pop()
    sidebyside = np.hstack(images)
    sidebyside = cv.normalize(sidebyside, None, 0, 255, cv.NORM_MINMAX)
    return sidebyside

def plot_voxel_grid(voxelgrid, cmap='gray'):
    sidebyside = get_voxel_grid_as_image(voxelgrid)
    plt.imshow(sidebyside, cmap=cmap)
    plt.show()

def voxel_grids_fixed_n_torch(xs, ys, ts, ps, B, n, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a list of voxel grids with a fixed number of events.
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    n : the number of events per voxel
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxels: list of output voxel grids
    """
    voxels = []
    for idx in range(0, len(xs)-n, n):
        voxels.append(events_to_voxel_torch(xs[idx:idx+n], ys[idx:idx+n],
            ts[idx:idx+n], ps[idx:idx+n], B, sensor_size=sensor_size,
            temporal_bilinear=temporal_bilinear))
    return voxels

def voxel_grids_fixed_t_torch(xs, ys, ts, ps, B, t, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a list of voxel grids with a fixed temporal width.
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    t : the time width of the voxel grids
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxels: list of output voxel grids
    """
    device = xs.device
    voxels = []
    np_ts = ts.cpu().numpy()
    for t_start in np.arange(ts[0].item(), ts[-1].item()-t, t):
        voxels.append(events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t_start, t_start+t, np_ts=np_ts,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear))
    return voxels

def events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t0, t1, device=None, np_ts=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a voxel grid of the events between t0 and t1
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    t0 : the start time of the voxel grid
    t1 : the end time of the voxel grid
    device : device to put voxel grid. If left empty, same device as events
    np_ts : a numpy copy of ts (optional). If not given, will be created in situ
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    assert(t1>t0)
    if np_ts is None:
        np_ts = ts.cpu().numpy()
    if device is None:
        device = xs.device
    start_idx = np.searchsorted(np_ts, t0)
    end_idx = np.searchsorted(np_ts, t1)
    assert(start_idx < end_idx)
    voxel = events_to_voxel_torch(xs[start_idx:end_idx], ys[start_idx:end_idx],
        ts[start_idx:end_idx], ps[start_idx:end_idx], B, device, sensor_size=sensor_size,
        temporal_bilinear=temporal_bilinear)
    return voxel

def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = t[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(t, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(t, 0, len(ts)-1, tend) 
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins

def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg

def events_to_voxel(xs, ys, ts, ps, B, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    num_events_per_bin = len(xs)//B
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = (np.expand_dims(np.zeros(t_norm.shape[0]), axis=0).transpose()).squeeze()
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = np.maximum(zeros, 1.0-np.abs(t_norm-bi))
            weights = ps*bilinear_weights
        else:
            beg = bi*num_events_per_bin
            end = beg + num_events_per_bin
            vb = events_to_image(xs[beg:end], ys[beg:end],
                    weights[beg:end], sensor_size=sensor_size)
        vb = events_to_image(xs, ys, weights.squeeze(), sensor_size=sensor_size, interpolation=None)
        bins.append(vb)
    bins = np.stack(bins)
    return bins

def events_to_neg_pos_voxel(xs, ys, ts, ps, B,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """
    pos_weights = np.where(ps, 1, 0)
    neg_weights = np.where(ps, 0, 1)

    voxel_pos = events_to_voxel(xs, ys, ts, pos_weights, B,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel(xs, ys, ts, neg_weights, B,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg
