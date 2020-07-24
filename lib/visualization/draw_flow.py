import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..util.event_util import clip_events_to_bounds
from ..util.util import flow2bgr_np
from .visualization_utils import *

def plot_flow_and_events(xs, ys, ts, ps, flow, save_path=None,
        num_compress='auto', num_show=1000,
        event_size=2, elev=0, azim=45, show_events=True,
        show_frames=True, show_plot=False, crop=None, compress_front=False,
        marker='.', stride = 20, invert=False, img_size=None, show_axes=False):

    #Crop events
    if img_size is None:
        img_size = [max(ys), max(ps)] if len(flow)==0 else flow[0].shape[1:3]
    crop = [0, 0, img_size[0], img_size[1]] if crop is None else crop
    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
    xs -= crop[1]
    ys -= crop[0]

    #Defaults and range checks
    num_show = len(xs) if num_show == -1 else num_show
    skip = max(len(xs)//num_show, 1)
    num_compress = len(xs) if num_compress == -1 else num_compress
    num_compress = min(img_size[0]*img_size[1]*0.5, len(xs)) if num_compress=='auto' else num_compress
    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]

    #Prepare the plot, set colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    colors = ['r' if p>0 else ('#00DAFF' if invert else 'b') for p in ps]

    img = flow2bgr_np(flow[0][0, :], flow[0][1, :])
    img = np.flip(np.flip(img/255, axis=0), axis=1)
    img = img[crop[0]:crop[2], crop[1]:crop[3], ...]
    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]

    print(len(xs))
    print(ts)
    ax.scatter(xs, ts, ys, zdir='z', c=colors, facecolors=colors,
            s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
    print("Scatter done")
    ax.plot_surface(y, ts[0], x, rstride=stride, cstride=stride, facecolors=img, alpha=1)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)

    plt.show()



def plot_between_frames(xs, ys, ts, ps, flows, flow_imgs, flow_ts, args, plttype='voxel'):
    args.crop = None if args.crop is None else parse_crop(args.crop)

    flow_event_idx = get_frame_indices(ts, flow_ts)
    if len(flow_ts.shape) == 1:
        flow_ts = frame_stamps_to_start_end(flow_ts)
        flow_event_idx = frame_stamps_to_start_end(flow_event_idx)
    prev_idx = 0
    for i in range(0, len(flows), args.skip_frames):
        flow = flows[i:i+args.skip_frames]
        flow_indices = flow_event_idx[i:i+args.skip_frames]
        s, e = flow_indices[-1,0], flow_indices[0,1]
        flow_ts = []
        for f_idx in flow_indices:
            flow_ts.append(ts[f_idx[1]])
        fname = os.path.join(args.output_path, "events_{:09d}.png".format(i))

        print("se: {}, {}".format(s, e))
        plot_flow_and_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], flow)
