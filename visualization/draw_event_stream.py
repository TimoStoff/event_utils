import argparse
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_formats.read_events import read_memmap_events, read_h5_events_dict
from representations.image import events_to_image
from util.event_util import clip_events_to_bounds
from tqdm import tqdm

def parse_crop(cropstr):
    split = cropstr.split("x")
    xsize = int(split[0])
    split = cropstr.split[1]("+")
    ysize = int(split[0])
    xoff = int(split[1])
    yoff = int(split[2])
    crop = [xoff, yoff, xoff+xsize, yoff+ysize]
    return crop

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating {directory}")
        os.makedirs(directory)

def combine_plotted(root_dir, elev=0, azim=45):
    if elev == 0 and azim == 45:
        pass

def plot_events_sliding(xs, ys, ts, ps, dt=None, sdt=None, save_dir="/tmp", frames=None, frame_ts=None, num_show=1000, event_size=2,
        skip_frames=5, show_skipped=True, elev=0, azim=0, show_events=True, show_frames=True, crop=None,
        compress_front=False, invert=False, num_compress='auto', show_plot=False, show_axes=False):
    if dt is None:
        dt = (ts[-1]-ts[0])/10
        sdt = dt/10
        print("Using dt={}, sdt={}".format(dt, sdt))
    if frames is not None:
        sensor_size = frames[0].shape
    else:
        sensor_size = [max(ys), max(xs)]

    if len(frame_ts.shape) == 2:
        frame_ts = frame_ts[:,1]
    for i, t0 in enumerate(tqdm(np.arange(ts[0], ts[-1]-dt, sdt))):
        te = t0+dt
        eidx0 = np.searchsorted(ts, t0)
        eidx1 = np.searchsorted(ts, te)
        fidx0 = np.searchsorted(frame_ts, t0)
        fidx1 = np.searchsorted(frame_ts, te)
        #print("{}:{} = {}".format(frame_ts[fidx0], ts[eidx0], fidx0))

        wxs, wys, wts, wps = xs[eidx0:eidx1], ys[eidx0:eidx1], ts[eidx0:eidx1], ps[eidx0:eidx1],
        if fidx0 == fidx1:
            wframes=[]
            wframe_ts=[]
        else:
            wframes = frames[fidx0:fidx1]
            wframe_ts = frame_ts[fidx0:fidx1]

        save_path = os.path.join(save_dir, "frame_{:010d}.png".format(i))
        plot_events(wxs, wys, wts, wps, save_path=save_path, num_show=num_show, event_size=event_size,
                imgs=wframes, img_ts=wframe_ts, show_events=show_events, azim=azim,
                elev=elev, show_frames=show_frames, crop=crop, compress_front=compress_front,
                invert=invert, num_compress=num_compress, show_plot=show_plot, img_size=sensor_size,
                show_axes=show_axes)

def plot_events(xs, ys, ts, ps, save_path=None, num_compress='auto', num_show=1000,
        event_size=2, elev=0, azim=45, imgs=[], img_ts=[], show_events=True,
        show_frames=True, show_plot=False, crop=None, compress_front=False,
        marker='.', stride = 10, invert=False, img_size=None, show_axes=False):
    """
    Given events, plot these in a spatiotemporal volume.
    :param: xs x coords of events
    :param: ys y coords of events
    :param: ts t coords of events
    :param: ps p coords of events
    :param: save_path if set, will save plot to here
    :param: num_compress will take this number of events from the end
        and create an event image from these. This event image will
        be displayed at the end of the spatiotemporal volume
    :param: num_show sets the number of events to plot. If set to -1
        will plot all of the events (can be potentially expensive)
    :param: event_size sets the size of the plotted events
    :param: elev sets the elevation of the plot
    :param: azim sets the azimuth of the plot
    :param: imgs a list of images to draw into the spatiotemporal volume
    :param: img_ts a list of the position on the temporal axis where each
        image from 'imgs' is to be placed (the timestamp of the images, usually)
    :param: show_events if False, will not plot the events (only images)
    :param: crop a list of length 4 that sets the crop of the plot (must
        be in the format [top_left_y, top_left_x, height, width]
    """
    #Crop events
    if img_size is None:
        img_size = [max(ys), max(ps)] if len(imgs)==0 else imgs[0].shape[0:2]
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
    #colors = ['r' if p>0 else '#00DAFF' for p in ps]
    colors = ['r' if p>0 else ('#00DAFF' if invert else 'b') for p in ps]

    #Plot images
    if len(imgs)>0 and show_frames:
        for img, img_ts in zip(imgs, img_ts):
            img = img[crop[0]:crop[2], crop[1]:crop[3]]
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

    ax.xaxis.set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if show_plot:
        plt.show()
    if save_path is not None:
        ensure_dir(save_path)
        plt.savefig(save_path, transparent=True, dpi=600)
    plt.close()

def plot_events_between_frames(xs, ys, ts, ps, frames, frame_event_idx, save_dir, num_show=1000, event_size=2,
        skip_frames=5, show_skipped=True, elev=0, azim=0, show_events=True, show_frames=True, crop=None,
        compress_front=False, invert=False, num_compress='auto', show_plot=False):
    prev_idx = 0
    for i in range(0, len(frames), skip_frames):
        if show_skipped:
            frame = frames[i:i+skip_frames]
            frame_indices = frame_event_idx[i:i+skip_frames]
        else:
            frame = [frames[i]]
            frame_indices = frame_event_idx[i][np.newaxis, ...]
        print("Processing frame {}".format(i))
        s, e = frame_indices[0,1], frame_indices[-1,0]
        img_ts = []
        for f_idx in frame_indices:
            img_ts.append(ts[f_idx[1]])
        fname = os.path.join(save_dir, "events_{:09d}.png".format(i))
        plot_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], save_path=fname, num_show=num_show, event_size=event_size,
                imgs=frame, img_ts=img_ts, show_events=show_events, azim=azim,
                elev=elev, show_frames=show_frames, crop=crop, compress_front=compress_front,
                invert=invert, num_compress=num_compress, show_plot=show_plot)

if __name__ == "__main__":
    """
    Quick demo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="memmap events path")
    parser.add_argument("--output_path", type=str, default="/tmp/visualization", help="Where to save image outputs")
    parser.add_argument('--show_plot', action='store_true', help='If true, will also display the plot in an interactive window.\
            Useful for selecting the desired orientation.')
    parser.add_argument("--num_show", type=int, default=-1, help="How many events to show per plot. If -1, show all events.")
    parser.add_argument("--event_size", type=float, default=2, help="Marker size of the plotted events")
    parser.add_argument("--elev", type=float, default=0, help="Elevation of plot")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth of plot")
    parser.add_argument("--skip_frames", type=int, default=1, help="Amount of frames to place per plot.")
    parser.add_argument("--start_frame", type=int, default=0, help="On which frame to start.")
    parser.add_argument('--hide_skipped', action='store_true', help='Do not draw skipped frames into plot.')
    parser.add_argument('--hide_events', action='store_true', help='Do not draw events')
    parser.add_argument('--hide_frames', action='store_true', help='Do not draw frames')
    parser.add_argument('--show_axes', action='store_true', help='Draw axes')
    parser.add_argument("--num_compress", type=int, default=0, help="How many events to draw compressed. If 'auto'\
            will automatically determine.", choices=['value', 'auto'])
    parser.add_argument('--compress_front', action='store_true', help='If set, will put the compressed events at the _start_\
            of the event volume, rather than the back.')
    parser.add_argument('--invert', action='store_true', help='If the figure is for a black background, you can invert the \
            colors for better visibility.')
    parser.add_argument("--crop", type=str, default=None, help="Set a crop of both images and events. Uses 'imagemagick' \
            syntax, eg for a crop of 10x20 starting from point 30,40 use: 10x20+30+40.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        events = read_memmap_events(args.path)

        ts = events['t'][:].squeeze()
        t0 = ts[0]
        ts = ts-t0
        frames = (events['images'][args.start_frame+1::])/255
        frame_idx = events['index'][args.start_frame::]
        frame_ts = events['frame_stamps'][args.start_frame+1::]-t0

        start_idx = np.searchsorted(ts, frame_ts[0])
        print("Starting from frame {}, event {}".format(args.start_frame, start_idx))

        xs = events['xy'][:,0]
        ys = events['xy'][:,1]
        ts = ts[:]
        ps = events['p'][:]

        print("Have {} frames".format(frames.shape))
    else:
        events = read_h5_events_dict(args.path)
        xs = events['xs']
        ys = events['ys']
        ts = events['ts']
        ps = events['ps']
        t0 = ts[0]
        ts = ts-t0
        frames = [np.flip(x/255., axis=0) for x in events['frames']]
        frame_ts = events['frame_timestamps'][1:]-t0
        frame_end = events['frame_event_indices'][1:]
        frame_start = np.concatenate((np.array([0]), frame_end))
        frame_idx = np.stack((frame_end, frame_start[0:-1]), axis=1)
        ys = frames[0].shape[0]-ys

    crop = None if args.crop is None else parse_crop(args.crop)
    #plot_events_between_frames(xs, ys, ts, ps, frames, frame_idx, args.output_path, num_show=args.num_show, event_size=args.event_size,
    #        skip_frames=args.skip_frames, show_skipped=not args.hide_skipped, azim=args.azim, elev=args.elev,
    #        show_events=not args.hide_events, show_frames=not args.hide_frames, crop=crop,
    #        num_compress=args.num_compress, compress_front=args.compress_front, invert=args.invert, show_plot=args.show_plot)
    plot_events_sliding(xs, ys, ts, ps, dt=None, sdt=None, frames=frames, frame_ts=frame_ts, save_dir=args.output_path,
            num_show=args.num_show, event_size=args.event_size,
            skip_frames=args.skip_frames, show_skipped=not args.hide_skipped, azim=args.azim, elev=args.elev,
            show_events=not args.hide_events, show_frames=not args.hide_frames, crop=crop,
            num_compress=args.num_compress, compress_front=args.compress_front, invert=args.invert,
            show_plot=args.show_plot, show_axes=args.show_axes)

