import numpy as np
import torch
import cv2 as cv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..util.event_util import clip_events_to_bounds
from ..util.util import flow2bgr_np
from ..transforms.optic_flow import warp_events_flow_torch
from ..representations.image import events_to_image_torch
from .visualization_utils import *

def motion_compensate(xs, ys, ts, ps, flow, fname="/tmp/img.png", crop=None):
    xs, ys, ts, ps, flow = torch.from_numpy(xs).type(torch.float32), torch.from_numpy(ys).type(torch.float32),\
        torch.from_numpy(ts).type(torch.float32), torch.from_numpy(ps).type(torch.float32), torch.from_numpy(flow).type(torch.float32)
    xw, yw = warp_events_flow_torch(xs, ys, ts, ps, flow)
    img_size = list(flow.shape)
    img_size.remove(2)
    img = events_to_image_torch(xw, yw, ps, sensor_size=img_size, interpolation='bilinear')
    img = np.flip(np.flip(img.numpy(), axis=0), axis=1)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    if crop is not None:
        img = img[crop[0]:crop[1], crop[2]: crop[3]]
    cv.imwrite(fname, img)

def plot_flow_and_events(xs, ys, ts, ps, flow, save_path=None,
        num_show=1000, event_size=2, elev=0, azim=45, show_events=True,
        show_frames=True, show_plot=False, crop=None,
        marker='.', stride = 20, img_size=None, show_axes=False,
        invert=False):

    print(event_size)
    #Crop events
    if img_size is None:
        img_size = [max(ys), max(xs)] if len(flow)==0 else flow[0].shape[1:3]
    crop = [0, img_size[0], 0, img_size[1]] if crop is None else crop
    xs, ys = img_size[1]-xs, img_size[0]-ys
    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
    xs -= crop[2]
    ys -= crop[0]
    img_size = [crop[1]-crop[0], crop[3]-crop[2]]
    xs, ys = img_size[1]-xs, img_size[0]-ys
    #flow[0] = flow[0][:, crop[0]:crop[1], crop[2]:crop[3]]
    flow = flow[0][:, crop[0]:crop[1], crop[2]:crop[3]]
    flow = np.flip(np.flip(flow, axis=1), axis=2)

    #Defaults and range checks
    num_show = len(xs) if num_show == -1 else num_show
    skip = max(len(xs)//num_show, 1)
    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]

    #Prepare the plot, set colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    colors = ['r' if p>0 else ('#00DAFF' if invert else 'b') for p in ps]

    # Plot quivers
    f_reshape = flow.transpose(1,2,0)
    print(f_reshape.shape)
    t_w = ts[-1]-ts[0]
    coords, flow_vals, magnitudes = [], [], []
    s = 20
    offset = 0
    thresh = 0
    print(img_size)
    for x in np.linspace(offset, img_size[1]-1-offset, s):
        for y in np.linspace(offset, img_size[0]-1-offset, s):
            ix, iy = int(x), int(y)
            flow_v = np.array([f_reshape[iy,ix,0]*t_w, f_reshape[iy,ix,1]*t_w, t_w])
            mag = np.linalg.norm(flow_v)
            if mag >= thresh:
                flow_vals.append(flow_v)
                magnitudes.append(mag)
                coords.append([x,y])
    magnitudes = np.array(magnitudes)
    max_flow = np.percentile(magnitudes, 99)

    x,y,z,u,v,w = [],[],[],[],[],[]
    idx = 0
    for coord, flow_vec, mag in zip(coords, flow_vals, magnitudes):
        #q_start = [coord[0], ts[0], coord[1]]
        rel_len = mag/max_flow
        flow_vec = flow_vec*rel_len
        x.append(coord[0])
        y.append(0.065)
        z.append(coord[1])
        u.append(max(1, flow_vec[0]))
        v.append(flow_vec[2])
        w.append(max(1, flow_vec[1]))
    ax.quiver(x,y,z,u,v,w,color='c', arrow_length_ratio=0, alpha=0.8)

    img = flow2bgr_np(flow[0, :], flow[1, :])
    img = img/255

    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    ax.plot_surface(y, ts[0], x, rstride=stride, cstride=stride, facecolors=img, alpha=1)

    ax.scatter(xs, ts, ys, zdir='z', c=colors, facecolors=colors,
            s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

    ax.view_init(elev=elev, azim=azim)

    ax.grid(False)
    # Hide panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if not show_axes:
        # Hide spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_frame_on(False)
    # Hide xy axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.show()



def plot_between_frames(xs, ys, ts, ps, flows, flow_imgs, flow_ts, args, plttype='voxel'):
    args.crop = None if args.crop is None else parse_crop(args.crop)

    flow_event_idx = get_frame_indices(ts, flow_ts)
    if len(flow_ts.shape) == 1:
        flow_ts = frame_stamps_to_start_end(flow_ts)
        flow_event_idx = frame_stamps_to_start_end(flow_event_idx)
    prev_idx = 0
    for i in range(0, len(flows), args.skip_frames):
        if i != 12:
            continue
        flow = flows[i:i+args.skip_frames]
        flow_indices = flow_event_idx[i:i+args.skip_frames]
        s, e = flow_indices[-1,0], flow_indices[0,1]

        motion_compensate(xs[s:e], ys[s:e], ts[s:e], ps[s:e], -np.flip(np.flip(flow[0], axis=1), axis=2).copy(), fname="/tmp/comp.png", crop=args.crop)
        motion_compensate(xs[s:e], ys[s:e], ts[s:e], ps[s:e], np.zeros_like(flow[0]), fname="/tmp/zero.png", crop=args.crop)
        e = np.searchsorted(ts, ts[s]+0.02)
        flow_ts = []
        for f_idx in flow_indices:
            flow_ts.append(ts[f_idx[1]])
        fname = os.path.join(args.output_path, "events_{:09d}.png".format(i))

        print("se: {}, {}".format(s, e))
        plot_flow_and_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], flow,
        num_show=args.num_show, event_size=args.event_size, elev=args.elev,
        azim=args.azim, show_events=not args.hide_events,
        show_frames=not args.hide_frames, show_plot=args.show_plot, crop=args.crop,
        stride=args.stride, show_axes=args.show_axes, invert=args.invert)
