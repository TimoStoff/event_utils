import numpy as np
import numpy.lib.recfunctions as nlr
import cv2 as cv
from skimage.measure import block_reduce
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..representations.image import events_to_image
from ..representations.voxel_grid import events_to_voxel
from ..util.event_util import clip_events_to_bounds
from .visualization_utils import *
from tqdm import tqdm

def plot_events_sliding(xs, ys, ts, ps, args, frames=[], frame_ts=[]):
    """
    Plot the given events in a sliding window fashion to generate a video
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param args Arguments for the rendering (see args list
        for 'plot_events' function)
    @param frames List of image frames
    @param frame_ts List of the image timestamps
    @returns None
    """
    dt, sdt = args.w_width, args.sw_width
    if dt is None:
        dt = (ts[-1]-ts[0])/10
        sdt = dt/10
        print("Using dt={}, sdt={}".format(dt, sdt))

    if len(frames) > 0:
        has_frames = True
        sensor_size = frames[0].shape
        frame_ts = frame_ts[:,1] if len(frame_ts.shape) == 2 else frame_ts
    else:
        has_frames = False
        sensor_size = [max(ys), max(xs)]

    n_frames = len(np.arange(ts[0], ts[-1]-dt, sdt))
    for i, t0 in enumerate(tqdm(np.arange(ts[0], ts[-1]-dt, sdt))):
        te = t0+dt
        eidx0 = np.searchsorted(ts, t0)
        eidx1 = np.searchsorted(ts, te)
        wxs, wys, wts, wps = xs[eidx0:eidx1], ys[eidx0:eidx1], ts[eidx0:eidx1], ps[eidx0:eidx1],

        wframes, wframe_ts = [], []
        if has_frames:
            fidx0 = np.searchsorted(frame_ts, t0)
            fidx1 = np.searchsorted(frame_ts, te)
            wframes = [frames[fidx0]]
            wframe_ts = [wts[0]]

        save_path = os.path.join(args.output_path, "frame_{:010d}.jpg".format(i))

        perc = i/n_frames
        min_p, max_p = 0.2, 0.7
        elev, azim = args.elev, args.azim
        max_elev, max_azim = 10, 45
        if perc > min_p and perc < max_p:
            p_way = (perc-min_p)/(max_p-min_p)
            elev = elev + (max_elev*p_way)
            azim = azim - (max_azim*p_way)
        elif perc >= max_p:
            elev, azim = max_elev, max_azim

        plot_events(wxs, wys, wts, wps, save_path=save_path, num_show=args.num_show, event_size=args.event_size,
                imgs=wframes, img_ts=wframe_ts, show_events=not args.hide_events, azim=azim,
                elev=elev, show_frames=not args.hide_frames, crop=args.crop, compress_front=args.compress_front,
                invert=args.invert, num_compress=args.num_compress, show_plot=args.show_plot, img_size=sensor_size,
                show_axes=args.show_axes, stride=args.stride)

def plot_voxel_grid(xs, ys, ts, ps, bins=5, frames=[], frame_ts=[],
        sensor_size=None, crop=None, elev=0, azim=45, show_axes=False):
    """
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param bins The number of bins to have in the voxel grid
    @param frames The list of image frames
    @param frame_ts The list of image timestamps
    @param sensor_size The size of the event sensor resolution
    @param crop Cropping parameters for the voxel grid (no crop if None)
    @param elev The elevation of the plot
    @param azim The azimuth of the plot
    @param show_axes Show the axes of the plot
    @returns None
    """
    if sensor_size is None:
        sensor_size = [np.max(ys)+1, np.max(xs)+1] if len(frames)==0 else frames[0].shape
    if crop is not None:
        xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop)
        sensor_size = crop_to_size(crop)
        xs, ys = xs-crop[2], ys-crop[0]
    num = 10000
    xs, ys, ts, ps = xs[0:num], ys[0:num], ts[0:num], ps[0:num]
    if len(xs) == 0:
        return
    voxels = events_to_voxel(xs, ys, ts, ps, bins, sensor_size=sensor_size)
    voxels = block_reduce(voxels, block_size=(1,10,10), func=np.mean, cval=0)
    dimdiff = voxels.shape[1]-voxels.shape[0]
    filler = np.zeros((dimdiff, *voxels.shape[1:]))
    voxels = np.concatenate((filler, voxels), axis=0)
    voxels = voxels.transpose(0,2,1)

    pltvoxels = voxels != 0
    pvp, nvp = voxels > 0, voxels < 0
    pvox, nvox = voxels*np.where(voxels > 0, 1, 0), voxels*np.where(voxels < 0, 1, 0)
    pvox, nvox = (pvox/np.max(pvox))*0.5+0.5, (np.abs(nvox)/np.max(np.abs(nvox)))*0.5+0.5
    zeros = np.zeros_like(voxels)

    colors = np.empty(voxels.shape, dtype=object)

    redvals = np.stack((pvox, zeros, pvox-0.5), axis=3)
    redvals = nlr.unstructured_to_structured(redvals).astype('O')

    bluvals = np.stack((nvox-0.5, zeros, nvox), axis=3)
    bluvals = nlr.unstructured_to_structured(bluvals).astype('O')

    colors[pvp] = redvals[pvp]
    colors[nvp] = bluvals[nvp]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(pltvoxels, facecolors=colors, edgecolor='k')
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

def plot_events(xs, ys, ts, ps, save_path=None, num_compress='auto', num_show=1000,
        event_size=2, elev=0, azim=45, imgs=[], img_ts=[], show_events=True,
        show_frames=True, show_plot=False, crop=None, compress_front=False,
        marker='.', stride = 1, invert=False, img_size=None, show_axes=False):
    """
    Given events, plot these in a spatiotemporal volume.
    @param xs x coords of events
    @param ys y coords of events
    @param ts t coords of events
    @param ps p coords of events
    @param save_path If set, will save plot to here
    @param num_compress Takes num_compress events from the beginning of the
        sequence and draws them in the plot at time $t=0$ in black
    @param compress_front If True, display the compressed events in black at the
        front of the spatiotemporal volume rather than the back
    @param num_show Sets the number of events to plot. If set to -1
        will plot all of the events (can be potentially expensive)
    @param event_size Sets the size of the plotted events
    @param elev Sets the elevation of the plot
    @param azim Sets the azimuth of the plot
    @param imgs A list of images to draw into the spatiotemporal volume
    @param img_ts A list of the position on the temporal axis where each
        image from 'imgs' is to be placed (the timestamp of the images, usually)
    @param show_events If False, will not plot the events (only images)
    @param show_plot If True, display the plot in a matplotlib window as
        well as saving to disk
    @param crop A list of length 4 that sets the crop of the plot (must
        be in the format [top_left_y, top_left_x, height, width]
    @param marker Which marker should be used to display the events (default
        is '.', which results in points, but circles 'o' or crosses 'x' are
        among many other possible options)
    @param stride Determines the pixel stride of the image rendering
        (1=full resolution, but can be quite resource intensive)
    @param invert Inverts the color scheme for black backgrounds
    @param img_size The size of the sensor resolution. Inferred if empty.
    @param show_axes If True, draw axes onto the plot.
    @returns None
    """
    #Crop events
    if img_size is None:
        img_size = [max(ys), max(xs)] if len(imgs)==0 else imgs[0].shape[0:2]
        print("Inferred image size = {}".format(img_size))
    crop = [0, img_size[0], 0, img_size[1]] if crop is None else crop
    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
    xs, ys = xs-crop[2], ys-crop[0]

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

    #Plot images
    if len(imgs)>0 and show_frames:
        for imgidx, (img, img_ts) in enumerate(zip(imgs, img_ts)):
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
            if len(img.shape)==2:
                img = np.stack((img, img, img), axis=2)
            if num_compress > 0:
                events_img = events_to_image(xs[0:num_compress], ys[0:num_compress],
                        np.ones(num_compress), sensor_size=img.shape[0:2])
                events_img[events_img>0] = 1
                img[:,:,1]+=events_img[:,:]
                img = np.clip(img, 0, 1)
            x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
            event_idx = np.searchsorted(ts, img_ts)

            ax.scatter(xs[0:event_idx], ts[0:event_idx], ys[0:event_idx], zdir='z',
                    c=colors[0:event_idx], facecolors=colors[0:event_idx],
                    s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

            ax.plot_surface(y, img_ts, x, rstride=stride, cstride=stride, facecolors=img, alpha=1)

            ax.scatter(xs[event_idx:-1], ts[event_idx:-1], ys[event_idx:-1], zdir='z',
                    c=colors[event_idx:-1], facecolors=colors[event_idx:-1],
                    s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

    elif num_compress > 0:
        # Plot events
        ax.scatter(xs[::skip], ts[::skip], ys[::skip], zdir='z', c=colors[::skip], facecolors=colors[::skip],
                s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
        num_compress = min(num_compress, len(xs))
        if not compress_front:
            ax.scatter(xs[0:num_compress], np.ones(num_compress)*ts[0], ys[0:num_compress],
                    marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
        else:
            ax.scatter(xs[-num_compress-1:-1], np.ones(num_compress)*ts[-1], ys[-num_compress-1:-1],
                    marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
    else:
        # Plot events
        ax.scatter(xs, ts, ys,zdir='z', c=colors, facecolors=colors, s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

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
    # Flush axes
    ax.set_xlim3d(0, img_size[1])
    ax.set_ylim3d(ts[0], ts[-1])
    ax.set_zlim3d(0,img_size[0])

    if show_plot:
        plt.show()
    if save_path is not None:
        ensure_dir(save_path)
        plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
    plt.close()

def plot_between_frames(xs, ys, ts, ps, frames, frame_event_idx, args, plttype='voxel'):
    """
    Plot events between frames for an entire sequence to form a video
    @param xs x component of events
    @param ys y component of events
    @param ts t component of events
    @param ps p component of events
    @param frames List of the frames
    @param frame_event_idx The event index for each frame
    @param args Arguments for the rendering function 'plot_events'
    @param plttype Whether to plot 'voxel' or 'events'
    @return None
    """
    args.crop = None if args.crop is None else parse_crop(args.crop)
    prev_idx = 0
    for i in range(0, len(frames), args.skip_frames):
        if args.hide_skipped:
            frame = [frames[i]]
            frame_indices = frame_event_idx[i][np.newaxis, ...]
        else:
            frame = frames[i:i+args.skip_frames]
            frame_indices = frame_event_idx[i:i+args.skip_frames]
        print("Processing frame {}".format(i))
        s, e = frame_indices[0,1], frame_indices[-1,0]
        img_ts = []
        for f_idx in frame_indices:
            img_ts.append(ts[f_idx[1]])
        fname = os.path.join(args.output_path, "events_{:09d}.png".format(i))
        if plttype == 'voxel':
            plot_voxel_grid(xs[s:e], ys[s:e], ts[s:e], ps[s:e], bins=args.num_bins, crop=args.crop,
                    frames=frame, frame_ts=img_ts, elev=args.elev, azim=args.azim)
        elif plttype == 'events':
            plot_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], save_path=fname,
                    num_show=args.num_show, event_size=args.event_size, imgs=frame,
                    img_ts=img_ts, show_events=not args.hide_events, azim=args.azim,
                    elev=args.elev, show_frames=not args.hide_frames, crop=args.crop,
                    compress_front=args.compress_front, invert=args.invert,
                    num_compress=args.num_compress, show_plot=args.show_plot, stride=args.stride)

